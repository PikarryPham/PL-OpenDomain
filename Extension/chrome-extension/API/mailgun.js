require('dotenv').config();
const formData = require('form-data');
const Mailgun = require('mailgun.js');

const domain = process.env.MAILGUN_DOMAIN

const mailgun = new Mailgun(formData);
const mg = mailgun.client({
  username: 'api',
  key: process.env.MAILGUN_API_KEY || 'key-yourkeyhere',
});

async function sendResetEmail(email, newPassword) {
  try {
    const response = await mg.messages.create(`${domain}`, {
      from: `Support Team <mailgun@${domain}>`,
      to: [email],
      subject: "Password Reset Request",
      text: `Your new password is: ${newPassword}`,
      html: `<p>Your new password is: <strong>${newPassword}</strong></p>`
    });

    console.log("Email sent successfully:", response);
    return true;
  } catch (error) {
    console.error("Error sending email:", error);
    return false;
  }
}

async function sendChangePassword(email) {
  try {
    const timestamp = new Date().toLocaleString("sv-SE").replace(" ", "T");
    const response = await mg.messages.create(`${domain}`, {
      from: `Support Team <mailgun@${domain}>`,
      to: [email],
      subject: "Change Password Request",
      text: `Your password was just changed at ${timestamp}.`,
      html: `<p>Your password was just changed at ${timestamp}.</p>`
    });

    console.log("Email sent successfully:", response);
    return true;
  } catch (error) {
    console.error("Error sending email:", error);
    return false;
  }
}

module.exports = { sendResetEmail, sendChangePassword };
