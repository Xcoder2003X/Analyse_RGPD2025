package com.example.pfa_uplaod.service;

import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;
import org.springframework.http.*;
import org.springframework.core.ParameterizedTypeReference;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.Map;

@Service
public class OpenAIService {

    private static final String OPENAI_API_URL = "https://openrouter.ai/api/v1/chat/completions";
    private static final String OPENAI_API_KEY = "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"; // 🔴
                                                                                                                              // Remplace
                                                                                                                              // par
                                                                                                                              // ta
                                                                                                                              // clé
                                                                                                                              // API
    private static final String RAG_SERVICE_URL = "http://localhost:5000/query";
    private final RestTemplate restTemplate;
    private static final Logger logger = LoggerFactory.getLogger(OpenAIService.class);

    public OpenAIService() {
        this.restTemplate = new RestTemplate();
    }

    private String getRAGContext(String question) {
        try {
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_JSON);

            Map<String, String> requestBody = Map.of("question", question);
            HttpEntity<Map<String, String>> request = new HttpEntity<>(requestBody, headers);

            ResponseEntity<Map<String, Object>> response = restTemplate.exchange(
                    RAG_SERVICE_URL,
                    HttpMethod.POST,
                    request,
                    new ParameterizedTypeReference<Map<String, Object>>() {
                    });

            Map<String, Object> responseBody = response.getBody();
            if (responseBody != null && responseBody.containsKey("answer")) {
                return (String) responseBody.get("answer");
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        return "";
    }

    public String analyzeDocument(String text) {
        logger.info("Starting document analysis");

        // Get relevant context from RAG
        String ragContext = getRAGContext(text);
        logger.info("Retrieved RAG context, length: {}", ragContext.length());

        String prompt = "Analyse ce texte et donne-moi un rapport détaillé sur : \n"
                + "- Qualité des données : Données manquantes, incohérentes ou non pertinentes.\n"
                + "- Diversité et représentativité : Statistiques sur les différentes catégories.\n"
                // + "- Détection des biais : Biais implicites ou distributions
                // déséquilibrées.\n"
                // + "- Transparence et traçabilité : Vérification de l'origine des
                // données.\n\n"
                + "est ce que le text a analyser est conforme aux regles du RGPD ? si non citer les points de non conformite et donner des recommandations pour la conformite totale"
                + "Contexte RGPD pertinent : \n" + ragContext + "\n\n"
                + "Voici le texte à analyser : \n" + text;

        logger.info("Sending request to OpenAI API");
        Map<String, Object> requestBody = Map.of(
                "model", "deepseek/deepseek-r1-zero:free",
                "messages", List.of(Map.of("role", "user", "content", prompt)),
                "temperature", 0.7);

        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        headers.setBearerAuth(OPENAI_API_KEY);

        HttpEntity<Map<String, Object>> request = new HttpEntity<>(requestBody, headers);

        try {
            ResponseEntity<Map<String, Object>> response = restTemplate.exchange(
                    OPENAI_API_URL,
                    HttpMethod.POST,
                    request,
                    new ParameterizedTypeReference<Map<String, Object>>() {
                    });

            Map<String, Object> responseBody = response.getBody();
            if (responseBody == null) {
                logger.error("Received null response from OpenAI API");
                throw new RuntimeException("Réponse vide de l'API OpenAI");
            }

            Object choicesObj = responseBody.get("choices");
            if (!(choicesObj instanceof List<?>)) {
                logger.error("Invalid choices format in OpenAI API response");
                throw new RuntimeException("Format de réponse OpenAI invalide");
            }

            @SuppressWarnings("unchecked")
            List<Map<String, Object>> choices = (List<Map<String, Object>>) choicesObj;
            if (choices.isEmpty()) {
                logger.error("No choices in OpenAI API response");
                throw new RuntimeException("Pas de choix dans la réponse de l'API OpenAI");
            }

            Map<String, Object> message = (Map<String, Object>) choices.get(0).get("message");
            String analysis = (String) message.get("content");
            logger.info("Successfully received analysis from OpenAI API");
            return analysis;
        } catch (Exception e) {
            logger.error("Error during OpenAI API call: {}", e.getMessage(), e);
            throw new RuntimeException("Erreur lors de l'appel à l'API OpenAI: " + e.getMessage());
        }
    }
}
