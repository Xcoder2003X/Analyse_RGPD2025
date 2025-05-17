import React, { useState } from "react";
import axios from "axios";
import "./signup.css";
import AuthService from "../services/AuthService";
import Robot3D from "./Robot3D";
import { keyframes } from "@emotion/react";
import styled from "@emotion/styled";
import { useTranslation } from "react-i18next";
import "../i18n"; // Ensure i18n is initialized

const fadeIn = keyframes`
    from { opacity: 0; }
    to { opacity: 1; }
  `;

  const AnimatedContainer = styled.div`
     position: relative;
    top: 70px;
    display: flex;
    align-items: center;
    gap: 40px;
    box-shadow: 4px 4px 10px 6px black;
    border-radius: 30px;
    padding: 30px;
    animation: ${fadeIn} 2s ease-out;
  `;
const SignUp = () => {
  const { t } = useTranslation();

  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [role, setRole] = useState("ROLE_USER"); // Default to "ROLE_USER"
  const currentUser = AuthService.getCurrentUser();
  const isAdmin = currentUser?.role === "ROLE_ADMIN";

  const handleSubmit = async (e) => {
    e.preventDefault();

    const signUpData = {
      username,
      password,
      role,
    };

    try {
      const response = await axios.post(
        "http://localhost/api/auth/signup",
        signUpData
      );
      alert(response.data);
    } catch (error) {
      console.error("Error during signup:", error);
      alert("There was an error during signup!");
    }
  };

  

  return (
    <AnimatedContainer>
      <Robot3D robotImage="robot.webP" />
      <form className="form" onSubmit={handleSubmit}>
        <p className="title">{t("register")}</p>
        <p className="message">{t("signupMessage")}</p>

        <label>
          <input
            className="input"
            type="text"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            required
          />
          <span>{t("username")}</span>
        </label>

        <label>
          <input
            className="input"
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
          />
          <span>{t("password")}</span>
        </label>
        <div style={{ display: "flex", gap: "20px", margin: "10px" }}>
          <label>{t("role")}:</label>
          <select
            value={role}
            onChange={(e) => setRole(e.target.value)}
            disabled={!isAdmin} // Disable the admin role selection for non-admin users
          >
            <option value="ROLE_USER">{t("user")}</option>
            {isAdmin && <option value="ROLE_ADMIN">{t("admin")}</option>}
          </select>
        </div>
        <button className="submit">{t("submit")}</button>
        <p className="signin">
          {t("alreadyHaveAccount")} <a href="login">{t("login")}</a>
        </p>
      </form>
    </AnimatedContainer>
  );
};

export default SignUp;
