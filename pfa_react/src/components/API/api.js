/*import axios from "axios";

export const fetchDashboard = async ({ pageNumber, token }) => {
    const response = await axios.get(
      `http://localhost:8080/api/reports/dashboard?page=${pageNumber}&size=8`,
      {
        headers: { Authorization: `Bearer ${token}` },
      }
    );
    return response.data;
  };
  
  export const analyzeReport = async ({ formData, token }) => {
    const response = await axios.post(
      "http://localhost:8080/api/reports/analyze",
      formData,
      {
        headers: { Authorization: `Bearer ${token}` },
      }
    );
    return response.data;
  };
  
  export const generatePdfReport = async (formData) => {
    const response = await axios.post(
      "http://localhost:8080/api/files/generate-report",
      formData,
      {
        responseType: "blob",
      }
    );
    return response.data;
  };*/