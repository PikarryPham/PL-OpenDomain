module.exports = {
    apps: [
      {
        name: "my-api",
        script: "api.js", // Đổi thành file chính của API
        instances: 4, // Chạy 4 instance
        exec_mode: "cluster", // Chạy ở chế độ cluster để load balancing
        env: {
          NODE_ENV: "development",
        },
        env_production: {
          NODE_ENV: "production",
        },
      },
    ],
  };
  