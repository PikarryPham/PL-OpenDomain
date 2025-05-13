# Web History Tracker Chrome Extension

A real-time privacy-focused Chrome extension that collects and analyzes user learning behaviors based on their browsing history within an open-domain learning context.

## 1. Features

- Automatically captures webpage visits with timestamps in real-time
- Stores page content (first 3000 characters) for better context
- Complete privacy (authentication + authorization + encryption password)
- Powerful search functionality across URLs, titles, page content, keywords
- Chronological sorting (newest/oldest/alphabet A-Z/alphabet Z-A/Time frame including Morning (6am-12pm), Afternoon (12pm-18pm), Evening (18pm-24am))
- Create account/Login/Change password/Logout
- Update account information & learning preferences
- etc....

## 2. Installation

### Client Side

1. Clone the code into your computer
2. Extract the ZIP file to a folder on your computer
3. Open Chrome and navigate to `chrome://extensions/`
4. Enable "Developer mode" in the top right corner
5. Click "Load unpacked" and select the extracted folder which named 'Chrome_Plugin_Base'
6. The extension icon should appear in your Chrome toolbar

### Server Side
1. Make sure the Clickhouse DB and Redpanda platform is working
2. Open new terminal and go to the folder /API
3. Enter the command ``` npm i ```
4. Enter the command ``` npm i -g pm2 ```
5. Enter the command ``` pm2 start ecosystem.config.js --env production ``` and make sure all processes are available
6. Enter the command ``` pm2 logs ``` 

## 3. Usage

### Note about .env file
In order to make the extension code run for the backend server, you need an .env file. However, we cannot push it into Github due to some reasons as follows:
1. It contains sensitive information, which if exposed can lead to unauthorized access or attacks.
2. It follows the principle of separating code from configuration, reducing risk when sharing or reviewing code.
3. It avoids being blocked by GitHub Secret Scanning, which would disrupt your workflow.
If you to run the code, please DM me on Github or contact me at trang.pham@jaist.ac.jp