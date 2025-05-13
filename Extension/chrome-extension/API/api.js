const express = require("express");
const cors = require('cors');
const bodyParser = require("body-parser");
const bcrypt = require("bcrypt");
const jwt = require("jsonwebtoken");
const { createClient } = require("@clickhouse/client");
const { v4: uuidv4 } = require("uuid");
const morgan = require("morgan");
require("dotenv").config();

const os = require("os")
const { Kafka, CompressionTypes } = require("kafkajs");
const e = require("express");

const { sendResetEmail, sendChangePassword } = require('./mailgun');

const app = express();
// add log middleware
app.use(morgan("dev"));
const PORT = process.env.PORT || 3000;
const SECRET_KEY = process.env.SECRET_KEY;

app.use(bodyParser.json());

const client = createClient({
    url: process.env.CLICKHOUSE_URL,
    username: process.env.CLICKHOUSE_USER,
    password: process.env.CLICKHOUSE_PASSWORD,
    database: process.env.CLICKHOUSE_DB,
    ssl: false,
});

const redpanda = new Kafka({
    brokers: [process.env.REDPANDA_BROKER],  // Thay bằng biến môi trường
    ssl: {}, 
    sasl: {
      mechanism: process.env.REDPANDA_MECHANISM, // "scram-sha-256" hoặc "scram-sha-512"
      username: process.env.REDPANDA_USERNAME,
      password: process.env.REDPANDA_PASSWORD
    }
});
const producer = redpanda.producer();
producer.connect();
const sendMessage = async (msg) => {
    try {
      await producer.send({
        topic: "history", // Đặt tên topic phù hợp
        compression: CompressionTypes.GZIP,
        messages: [{
          key: os.hostname(),
          value: JSON.stringify(msg)  // Chuyển đổi object thành JSON string
        }]
      });
      console.log("Message sent successfully to Redpanda");
    } catch (e) {
      console.error(`Unable to send message: ${e.message}`, e);
    }
};
  
// Middleware xác thực token
function authenticateToken(req, res, next) {
    const token = req.header("Authorization");
    if (!token) return res.status(401).json({ success: false, message: "Không có token!" });

    jwt.verify(token, SECRET_KEY, (err, user) => {
        if (err) return res.status(403).json({ success: false, message: "Token không hợp lệ!" });
        req.user = user;
        next();
    });
}

// Regex kiểm tra username (chỉ chứa chữ cái, số, dấu gạch dưới, tối đa 200 ký tự)
const usernameRegex = /^[a-zA-Z0-9_]{1,200}$/;
const emailRegex = /^[a-zA-Z0-9._-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,6}$/;

app.use(cors({
    origin: 'http://127.0.0.1:5500' // Or use '*' to allow all origins
  }));

// API kiểm tra email
app.post("/check_email", async (req, res) => {
    const { email } = req.body;

    if (!email) {
        return res.status(400).json({ success: false, message: "No email" });
    }

    try {
        const query = `SELECT * FROM users WHERE email = {email:String}`;
        const result = await client.query({ query: query, query_params: { email }, format: "JSONEachRow" });
        const user = await result.json();

        if (user.length > 0) {
            return res.status(404).json({ success: false, message: "Email already exists" });
        }

        res.json({ success: true, message: "Valid email" });
    } catch (error) {
        console.error("Lỗi kiểm tra email:", error);
        res.status(500).json({ success: false, message: "Lỗi máy chủ" });
    }
});

// API đăng ký tạm thời (để lưu thông tin đăng ký khi chưa hoàn tất khảo sát)
app.post("/temp_register", async (req, res) => {
    const { username, email, password } = req.body;

    if (!username || !email || !password) {
        return res.status(400).json({ success: false, message: "Please enter complete information!" });
    }

    if (!usernameRegex.test(username)) {
        return res.status(400).json({ success: false, message: "Invalid username!" });
    }

    if (!emailRegex.test(email)) {
        return res.status(400).json({ success: false, message: "Invalid email!" });
    }

    if (password.length < 6) {
        return res.status(400).json({ success: false, message: "Password must be at least 6 characters!" });
    }

    try {
        const hashedPassword = await bcrypt.hash(password, 10);
        const created_at = new Date().toISOString().slice(0, 19).replace("T", " ");

        // Kiểm tra xem email đã tồn tại trong bảng temp_users chưa
        const checkQuery = `SELECT id FROM temp_users WHERE email = {email:String} LIMIT 1`;
        const checkResult = await client.query({ query: checkQuery, query_params: { email }, format: "JSONEachRow" });
        const existingUser = await checkResult.json();

        if (existingUser.length > 0) {
            // Nếu đã tồn tại, cập nhật thông tin thay vì tạo mới
            const updateQuery = `
                ALTER TABLE temp_users UPDATE 
                username = {username:String}, 
                password = {password:String}, 
                created_at = {created_at:DateTime} 
                WHERE email = {email:String}
            `;

            await client.command({
                query: updateQuery,
                query_params: { username, email, password: hashedPassword, created_at }
            });

            res.json({ success: true, temp_id: existingUser[0].id, message: "Registration information has been updated!" });
        } else {
            // Nếu chưa tồn tại, tạo một bản ghi mới
            const tempId = crypto.randomUUID();
            const insertQuery = `
                INSERT INTO temp_users (id, username, email, password, created_at) 
                VALUES ({id:UUID}, {username:String}, {email:String}, {password:String}, {created_at:DateTime})
            `;

            await client.command({
                query: insertQuery,
                query_params: { id: tempId, username, email, password: hashedPassword, created_at }
            });

            res.json({ success: true, temp_id: tempId, message: "Registration information is temporarily saved!" });
        }
    } catch (error) {
        console.error("Lỗi lưu tạm thông tin đăng ký:", error);
        res.status(500).json({ success: false, message: "Lỗi máy chủ!" });
    }
});

app.post("/final_register", async (req, res) => {
    const { temp_id, selections } = req.body; // Lấy temp_id và selections từ request body

    if (!temp_id || !selections || selections.length === 0) {
        return res.status(400).json({ success: false, message: "Incomplete information!" });
    }

    // Kiểm tra tính hợp lệ của selections
    if (!Array.isArray(selections)) {
        return res.status(400).json({ success: false, message: "selections must be an array!" });
    }

    // Kiểm tra từng phần tử của selections
    const invalidSelections = selections.filter(selection => 
        !selection.hasOwnProperty("form") || 
        typeof selection.form !== "string" ||
        !selection.hasOwnProperty("option") || 
        !(Array.isArray(selection.option) && 
        selection.option.every(opt => 
            typeof opt === "object" && 
            opt.hasOwnProperty("order") && 
            Number.isInteger(opt.order) && 
            opt.hasOwnProperty("option") && 
            typeof opt.option === "string"
        )
        )
    );

    if (invalidSelections.length > 0) {
        return res.status(400).json({ success: false, message: "selections contains invalid data!" });
    }


    // Khởi tạo các trường với mảng trống
    let preferred_areas = [];
    let preferred_content_types = [];
    let preferred_learn_style = [];
    let education_lv = [];

    try {
        // Lấy thông tin từ temp_users bằng temp_id
        const selectQuery = `SELECT * FROM temp_users WHERE id = {temp_id:UUID}`;
        const result = await client.query({ query: selectQuery, query_params: { temp_id }, format: "JSONEachRow" });
        const tempUser = await result.json();

        if (tempUser.length === 0) {
            return res.status(404).json({ success: false, message: "User information not found!" });
        }

        const { username, email, password } = tempUser[0];
        const hashedPassword = password;
        const created_at = new Date().toLocaleString("sv-SE").replace(" ", "T");

        // Function xử lý định dạng dữ liệu
        const formatArrayForClickHouse = (array) => {
            return JSON.stringify(array); // Chuyển thành JSON hợp lệ
        };

        // Duyệt qua selections để gán số thứ tự cho từng option
        selections.forEach(selection => {
            const orderedOptions = selection.option.map(opt => [opt.order, opt.option]);

            if (selection.form === 'preferred_areas') {
                preferred_areas = orderedOptions;
            } else if (selection.form === 'preferred_content_types') {
                preferred_content_types = orderedOptions;
            } else if (selection.form === 'preferred_learn_style') {
                preferred_learn_style = orderedOptions;
            } else if (selection.form === 'education_lv') {
                education_lv = orderedOptions;
            }
        });
        
        // Chuyển dữ liệu thành JSON đúng định dạng
        preferred_areas = formatArrayForClickHouse(preferred_areas);
        preferred_content_types = formatArrayForClickHouse(preferred_content_types);
        preferred_learn_style = formatArrayForClickHouse(preferred_learn_style);
        education_lv = formatArrayForClickHouse(education_lv);
        

        // Debug kiểm tra dữ liệu trước khi chạy query
        console.log("Formatted preferred_areas:", preferred_areas);


        
        const insertQuery = `
            INSERT INTO default.users 
            (username, email, password, created_time, updated_time, preferred_areas, preferred_content_types, preferred_learn_style, education_lv) 
            VALUES 
            ('${username}', '${email}', '${hashedPassword}', '${created_at}', '${created_at}', 
            JSONExtract('${preferred_areas}', 'Array(Tuple(Int32, String))'),
            JSONExtract('${preferred_content_types}', 'Array(Tuple(Int32, String))'),
            JSONExtract('${preferred_learn_style}', 'Array(Tuple(Int32, String))'),
            JSONExtract('${education_lv}', 'Array(Tuple(Int32, String))'))
        `;

        await client.command({ query: insertQuery });


        // Xóa thông tin người dùng khỏi bảng temp_users
        const deleteQuery = `DELETE FROM temp_users WHERE id = {temp_id:UUID}`;
        await client.command({
            query: deleteQuery,
            query_params: { temp_id }
        });

        res.json({ success: true, message: "Registration is complete and information has been saved to the users table.!" });
    } catch (error) {
        console.error("Lỗi lưu thông tin vào bảng users:", error);
        res.status(500).json({ success: false, message: "Lỗi máy chủ!" });
    }
});

// API đăng nhập
app.post("/login", async (req, res) => {
    const { email, password } = req.body;

    if (!email || !password) {
        return res.status(400).json({ success: false, message: "Email and password cannot be blank!" });
    }

    try {
        const query = `SELECT user_id, email, password FROM users WHERE email = {email:String} LIMIT 1`;
        const resultSet = await client.query({
            query,
            query_params: { email }, // Truyền email vào tham số
            format: "JSONEachRow"
        });

        const users = await resultSet.json();

        if (users.length === 0) {
            return res.status(401).json({ success: false, message: "Email is incorrect.!" });
        }

        const user = users[0];
        const passwordMatch = await bcrypt.compare(password, user.password);

        if (!passwordMatch) {
            return res.status(401).json({ success: false, message: "Incorrect password!" });
        }

        const token = jwt.sign({ id: user.user_id }, SECRET_KEY, { expiresIn: "8h" });
        res.json({ success: true, token });
    } catch (error) {
        console.error("Lỗi truy vấn ClickHouse:", error);
        res.status(500).json({ success: false, message: "Incorrect email and password combination!" });
    }
});

// API thêm câu hỏi 
app.post("/add_question", async (req, res) => {
    const { question_text, type, options } = req.body;

    if (!question_text || !type || !Array.isArray(options) || options.length === 0) {
        return res.status(400).json({ success: false, message: "Dữ liệu không hợp lệ!" });
    }

    try {
        // Tạo UUID cho câu hỏi
        const questionId = crypto.randomUUID();
        const created_at = new Date().toISOString().slice(0, 19).replace("T", " ");

        // Chèn câu hỏi vào bảng `questions`
        const insertQuestionQuery = `
            INSERT INTO questions (id, question_text, type, created_at) VALUES
            ({id:UUID}, {question_text:String}, {type:String}, {created_at:DateTime})
        `;

        await client.command({
            query: insertQuestionQuery,
            query_params: { id: questionId, question_text, type, created_at }
        });

        // Chèn các tùy chọn vào bảng `options`
        const insertOptionsQuery = `
            INSERT INTO options (id, question_id, option_text) VALUES
        ` + options.map((_, i) => `({id${i}:UUID}, {question_id:UUID}, {option_text${i}:String})`).join(", ");

        const query_params = { question_id: questionId };
        options.forEach((option, i) => {
            query_params[`id${i}`] = crypto.randomUUID();
            query_params[`option_text${i}`] = option;
        });

        await client.command({ query: insertOptionsQuery, query_params });

        res.json({ success: true, message: "Thêm câu hỏi thành công!" });
    } catch (error) {
        console.error("Lỗi thêm câu hỏi:", error);
        res.status(500).json({ success: false, message: "Lỗi máy chủ!" });
    }
});

// API lấy danh sách câu hỏi
app.get("/get_questions", async (req, res) => {
    try {
        // Truy vấn lấy câu hỏi và tùy chọn cùng một lúc
        const query = `
            SELECT q.id, q.question_text, q.type, q.form, o.option_text
            FROM questions q
            LEFT JOIN options o ON q.id = o.question_id
        `;
        const result = await client.query({ query, format: "JSONEachRow" });
        const questions = await result.json();

        // Gộp câu hỏi với tùy chọn
        const response = [];

        // Gộp theo id câu hỏi và tạo mảng options cho mỗi câu hỏi
        questions.forEach(q => {
            let question = response.find(item => item.id === q.id);

            if (!question) {
                question = {
                    id: q.id,
                    question_text: q.question_text,
                    type: q.type,
                    form: q.form,
                    options: []
                };
                response.push(question);
            }

            if (q.option_text) {
                question.options.push(q.option_text);
            }
        });

        // Trả về kết quả
        res.json({ success: true, questions: response });
    } catch (error) {
        console.error("Lỗi lấy danh sách câu hỏi:", error);
        res.status(500).json({ success: false, message: "Lỗi máy chủ!" });
    }
});

// API lấy thông tin người dùng
app.get("/user_info", authenticateToken, async (req, res) => {
    try {
        const query = `SELECT * FROM users WHERE email = {email:String} LIMIT 1`;
        const resultSet = await client.query({
            query,
            query_params: { email: req.user.email },
            format: "JSONEachRow"
        });

        const users = await resultSet.json();

        if (users.length === 0) {
            return res.status(404).json({ success: false, message: "Người dùng không tồn tại!" });
        }

        res.json({ success: true, user: users[0] });
    } catch (error) {
        console.error("Lỗi lấy thông tin người dùng:", error);
        res.status(500).json({ success: false, message: "Lỗi máy chủ!" });
    }
});

// API lưu lịch sử
app.post("/save_history", authenticateToken, async (req, res) => {
    try {

        console.log(req.body.title);
        // get page view count
        let oldPageView = 0;
        const query = `SELECT count(*) as count FROM browsinghistory WHERE url = {url:String} AND user_id = {user_id:UUID} LIMIT 1 `;
        const resultSet = await client.query({ query, query_params: { url: req.body.url, user_id: req.user.id } });
        const pageviews = await resultSet.json();
        if (pageviews.data.length > 0) {
            oldPageView =  pageviews.data[0].count
        }
        // ----
        const timestamp = new Date().toLocaleString("sv-SE").replace(" ", "T");




        // Lấy các trường từ request body
        const { 
            title, 
            visible_content, 
            url, 
            referrer_page, 
            exit_page,
            browser_id,
        } = req.body;

        // save tmp keyword
        const tmp_keywords = extractKeywords(title, url);
        const search_keyword = extractSearchKeyword(url);

        const historyData = {
            entry_id: uuidv4(),
            user_id: req.user.id, 
            title, 
            visible_content, 
            tmp_keywords, 
            timestamp: timestamp, 
            url, 
            referrer_page, 
            exit_page, 
            pageview_count: Number(oldPageView) + 1, 
            search_keyword,
            browser_id
        };

        // Gửi dữ liệu vào Redpanda (Kafka)
        await sendMessage(historyData);

        res.json({ success: true, message: "History data sent to Redpanda" });
    } catch (error) {
        console.error("Lỗi lưu history:", error);
        res.status(500).json({ success: false, message: error.message });
    }
});

app.post("/exit_page", authenticateToken, async (req, res) => {
    try {
        const { url, browser_id } = req.body;


        const updateQuery = `ALTER TABLE browsinghistory UPDATE exit_page = {url:String} WHERE browser_id = {browser_id:String}`;
    
        await client.command({
            query: updateQuery,
            query_params: { url: url, browser_id: browser_id}
        });
        res.json({ success: true, message: "Exit page updated successfully!" });
    }
    catch (error) {
        console.error("Lỗi cập nhật trang thoát:", error);
        res.status(500).json({ success: false, message: "Lỗi máy chủ!" });
    }
})
// API lấy lịch sử theo user_id
app.get("/get_history", authenticateToken, async (req, res) => {
  try {
    const query = `SELECT * FROM browsinghistory WHERE user_id = {user_id:UUID} ORDER BY timestamp DESC`;
    const resultSet = await client.query({ query, query_params: { user_id: req.user.id } });
    const history = await resultSet.json();

    res.json({ success: true, history });
  } catch (error) {
    res.status(500).json({ success: false, message: "Lỗi máy chủ!" });
  }
});

// API lấy thông tin người dùng theo user_id
app.get("/get_account", authenticateToken, async (req, res) => {
    try {
        const query = `SELECT * FROM users WHERE user_id = {user_id:UUID} LIMIT 1`;
        const resultSet = await client.query({ query, query_params: { user_id: req.user.id } });
        const users = await resultSet.json();

        if (users.length === 0) {
            return res.status(404).json({ success: false, message: "Người dùng không tồn tại!" });
        }

        res.json({ success: true, user: users.data[0] });
    } catch (error) {
        res.status(500).json({ success: false, message: "Lỗi máy chủ!" });
    }
});

// API đổi mật khẩu
app.post("/change_password", authenticateToken, async (req, res) => {
    const { email, old_password, new_password } = req.body;

    if (new_password.length < 6) {
        return res.status(400).json({ success: false, message: "Password must be at least 6 characters!" });
    }

    if (old_password == new_password) {
        return res.status(400).json({ success: false, message: "New password cannot be the same as old password!" });
    }

    try {
        const query = `SELECT password FROM users WHERE user_id = {user_id:UUID} LIMIT 1`;
        const resultSet = await client.query({ query, query_params: { user_id: req.user.id } });
        const users = await resultSet.json();

        if (users.length === 0) {
            return res.status(404).json({ success: false, message: "User does not exist!" });
        }

        const user = users.data[0];
        const passwordMatch = await bcrypt.compare(old_password, user.password);

        if (!passwordMatch) {
            return res.status(401).json({ success: false, message: "Old password is incorrect!" });
        }

        const hashedPassword = await bcrypt.hash(new_password, 10);
        const updateQuery = `ALTER TABLE users UPDATE password = {password:String}, updated_time = {updated_time:DateTime} WHERE user_id = {user_id:UUID}`;
        const updateDate = new Date().toISOString().slice(0, 19).replace("T", " ");

        await client.command({
            query: updateQuery,
            query_params: { user_id: req.user.id, password: hashedPassword, updated_time: updateDate }
        });

        await sendChangePassword(email);

        res.json({ success: true, message: "Password changed successfully!" });
    } catch (error) {
        console.error("Lỗi đổi mật khẩu:", error);
        res.status(500).json({ success: false, message: "Lỗi máy chủ!" });
    }
});

// API đổi thông tin khảo sát
app.post("/change_survey", authenticateToken, async (req, res) => {
    const { selections } = req.body; // Nhận selections từ request body
    const created_at = new Date().toLocaleString("sv-SE").replace(" ", "T");


    if (!selections || selections.length === 0) {
        return res.status(400).json({ success: false, message: "Incomplete information!" });
    }

    if (!Array.isArray(selections)) {
        return res.status(400).json({ success: false, message: "selections must be an array!" });
    }

    // Kiểm tra từng phần tử của selections
    const invalidSelections = selections.filter(selection => 
        !selection.hasOwnProperty("form") || 
        typeof selection.form !== "string" ||
        !selection.hasOwnProperty("option") || 
        !(Array.isArray(selection.option) && 
        selection.option.every(opt => 
            typeof opt === "object" && 
            opt.hasOwnProperty("order") && 
            Number.isInteger(opt.order) && 
            opt.hasOwnProperty("option") && 
            typeof opt.option === "string"
        )
        )
    );

    if (invalidSelections.length > 0) {
        return res.status(400).json({ success: false, message: "selections contains invalid data!" });
    }

    let preferred_areas = [];
    let preferred_content_types = [];
    let preferred_learn_style = [];
    let education_lv = [];

    try {
        // Duyệt qua selections để gán số thứ tự cho từng option
        selections.forEach(selection => {
            const orderedOptions = selection.option.map(opt => [opt.order, opt.option]);

            if (selection.form === 'preferred_areas') {
                preferred_areas = orderedOptions;
            } else if (selection.form === 'preferred_content_types') {
                preferred_content_types = orderedOptions;
            } else if (selection.form === 'preferred_learn_style') {
                preferred_learn_style = orderedOptions;
            } else if (selection.form === 'education_lv') {
                education_lv = orderedOptions;
            }
        });

        // Chuyển đổi dữ liệu thành JSON hợp lệ
        const formatArrayForClickHouse = (array) => JSON.stringify(array);

        preferred_areas = formatArrayForClickHouse(preferred_areas);
        preferred_content_types = formatArrayForClickHouse(preferred_content_types);
        preferred_learn_style = formatArrayForClickHouse(preferred_learn_style);
        education_lv = formatArrayForClickHouse(education_lv);

        // Câu lệnh UPDATE
        const updateQuery = `
            ALTER TABLE users UPDATE 
            updated_time = '${created_at}',
            preferred_areas = JSONExtract('${preferred_areas}', 'Array(Tuple(Int32, String))'),
            preferred_content_types = JSONExtract('${preferred_content_types}', 'Array(Tuple(Int32, String))'),
            preferred_learn_style = JSONExtract('${preferred_learn_style}', 'Array(Tuple(Int32, String))'),
            education_lv = JSONExtract('${education_lv}', 'Array(Tuple(Int32, String))')
            WHERE user_id = {user_id:UUID}
        `;

        await client.command({
            query: updateQuery,
            query_params: { user_id: req.user.id }
        });

        res.json({ success: true, message: "Survey information updated successfully!" });
    } catch (error) {
        console.error("Lỗi cập nhật thông tin khảo sát:", error);
        res.status(500).json({ success: false, message: "Lỗi máy chủ!" });
    }
});

// API reset mật khẩu
app.post("/reset_password", async (req, res) => {
    const { email } = req.body;
    const created_at = new Date().toLocaleString("sv-SE").replace(" ", "T");

    try {
        const query = `SELECT user_id FROM users WHERE email = {email:String} LIMIT 1`;
        const resultSet = await client.query({ query, query_params: { email } });
        const users = await resultSet.json();

        if (users.length === 0) {
            return res.status(404).json({ success: false, message: "Email không tồn tại!" });
        }

        const user = users.data[0];
        const newPassword = crypto.randomUUID().substring(0, 8);
        const hashedPassword = await bcrypt.hash(newPassword, 10);

        const updateQuery = `ALTER TABLE users UPDATE updated_time = {updated_time: DateTime}, password = {password:String} WHERE user_id = {user_id:UUID}`;
        await client.command({ query: updateQuery, query_params: { user_id: user.user_id, updated_time: created_at, password: hashedPassword } });

        await sendResetEmail(email, newPassword);

        res.json({ success: true, message: "Mật khẩu mới đã được gửi vào email của bạn!" });
    } catch (error) {
        res.status(500).json({ success: false, message: "Lỗi máy chủ!" });
    }
});

// Hàm lấy từ khóa từ nội dung trang
function extractKeywords(title, url) {
    // Function to extract keywords from the title
    function extractTitleKeywords(title) {
        return title
            .replace(/[^a-zA-Z0-9\s]/g, "") // Remove special characters
            .split(/\s+/) // Split by whitespace
            .filter(word => word.length > 1); // Remove single-letter words
    }
  
    // Function to extract keywords from the URL
    function extractURLKeywords(url) {
        const urlObj = new URL(url);
        const searchParams = ["q", "query", "search", "keyword", "term", "p", "s", "text", "searchtext", "kw", "key"];
        const pathPatterns = [/\/search\/([^\/]+)/, /\/find\/([^\/]+)/, /\/search-results\/([^\/]+)/, /\/keywords\/([^\/]+)/, /\/tag\/([^\/]+)/];
        
        // Check query parameters
        for (const param of searchParams) {
            if (urlObj.searchParams.has(param)) {
                return urlObj.searchParams.get(param).split("+").map(decodeURIComponent);
            }
        }
  
        // Check path patterns
        for (const pattern of pathPatterns) {
            const match = urlObj.pathname.match(pattern);
            if (match) {
                return match[1].split("-").map(decodeURIComponent);
            }
        }
        return [];
    }
  
    // Extract keywords from title and URL
    const titleKeywords = extractTitleKeywords(title);
    const urlKeywords = extractURLKeywords(url);
    
    // Combine and remove duplicates
    return [...new Set([...titleKeywords, ...urlKeywords])];
  }
  
  // Trích xuất từ khóa tìm kiếm từ URL
  function extractSearchKeyword(pageUrl) {
      try {
          const parsedUrl = new URL(pageUrl);
          const searchParams = ["q", "query", "search", "search_query", "keyword", "term", "text", "p", "s", "searchtext", "kw", "key"];
  
          for (const param of searchParams) {
              if (parsedUrl.searchParams.has(param)) {
                  return decodeURIComponent(parsedUrl.searchParams.get(param).replace(/\+/g, " "));
              }
          }
          return null;
      } catch (error) {
          console.error("Error processing URL:", error);
          return null;
      }
  }
  
// API kiểm tra pageview_count của trang dựa vào url và user_id

app.listen(PORT, () => {
    console.log(`Server is running on port ${PORT}`);
});
