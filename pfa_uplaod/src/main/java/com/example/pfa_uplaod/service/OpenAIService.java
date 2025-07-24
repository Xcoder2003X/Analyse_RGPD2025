package com.example.pfa_uplaod.service;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;
import org.springframework.http.*;
import org.springframework.core.ParameterizedTypeReference;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

@Service
public class OpenAIService {    private static final Logger logger = LoggerFactory.getLogger(OpenAIService.class);

    @Value("${openai.api.url}")
    private String OPENAI_API_URL;

    @Value("${openai.api.key}")
    private String OPENAI_API_KEY;

    @Value("${rag.service.url}")
    private String RAG_SERVICE_URL;

    private static final String FRONTEND_URL = "http://localhost";
    private static final String APP_USER_AGENT = "RGPD-Analyzer/1.0.0";

    private final RestTemplate restTemplate;

    public OpenAIService() {
        this.restTemplate = new RestTemplate();
    }

    /**
     * Get relevant context from RAG service.
     */
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
            logger.error("Error retrieving RAG context: {}", e.getMessage(), e);
        }
        return "";
    }

    /**
     * Analyze the document for a detailed report.
     */
    public String analyzeDocument(String text) {
        logger.info("Starting document analysis");

        String ragContext = getRAGContext(text);
        logger.info("Retrieved RAG context, length: {}", ragContext.length());

        String prompt = "Analyse ce texte et donne-moi un rapport détaillé sur : \n" +
                "- Qualité des données : Données manquantes, incohérentes ou non pertinentes.\n" +
                "- Diversité et représentativité : Statistiques sur les différentes catégories.\n" +
                "Est-ce que le texte à analyser est conforme aux règles du RGPD ? " +
                "Si non, citer les points de non-conformité et donner des recommandations pour la conformité totale.\n"
                +
                "Contexte RGPD pertinent : \n" + ragContext + "\n\n" +
                "Voici le texte à analyser : \n" + text;        
                logger.info("Sending request to OpenAI API with prompt: {}", prompt);        Map<String, Object> requestBody = new HashMap<>();
                requestBody.put("model", "deepseek/deepseek-r1:free");
                requestBody.put("messages", List.of(Map.of("role", "user", "content", prompt)));
                requestBody.put("temperature", 0.7);
                requestBody.put("max_tokens", 2000);
        
        HttpHeaders headers = createOpenRouterHeaders();
        HttpEntity<Map<String, Object>> request = new HttpEntity<>(requestBody, headers);

        try {
            ResponseEntity<Map<String, Object>> response = restTemplate.exchange(
                    OPENAI_API_URL,
                    HttpMethod.POST,
                    request,
                    new ParameterizedTypeReference<Map<String, Object>>() {
                    });

            Map<String, Object> responseBody = response.getBody();
            logger.info("OpenAI API response body: {}", responseBody);

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

            @SuppressWarnings("unchecked")
            Map<String, Object> message = (Map<String, Object>) choices.get(0).get("message");
            String analysis = (String) message.get("content");
            logger.info("Successfully received analysis from OpenAI API: {}", analysis);
            return analysis;
        } catch (Exception e) {
            logger.error("Error during OpenAI API call: {}", e.getMessage(), e);
            throw new RuntimeException("Erreur lors de l'appel à l'API OpenAI: " + e.getMessage());
        }
    }

    /**
     * Check the RGPD conformity using a dedicated prompt.
     */
    public Map<String, Object> checkConformity(String text) {
        logger.info("Starting conformity check");
        String ragContext = getRAGContext(text);        String prompt = "Analyse la conformité RGPD de ce texte et réponds avec :\n" +
                "[SCORE:] (nombre entre 0-100)\n" +
                "[CONSENTEMENT:] (OUI/NON)\n" +
                "[RAISONS:] (liste concise des raisons, en indiquant 'non' pour les points non conformes a RGPD et 'oui' pour les points conformes )\n\n" +
                "Contexte RGPD : \n" + ragContext + "\n\n" +
                "Texte à analyser :\n" + text;        Map<String, Object> requestBody = new HashMap<>();
        requestBody.put("model", "deepseek/deepseek-r1:free");
        requestBody.put("messages", List.of(Map.of("role", "user", "content", prompt)));
        requestBody.put("temperature", 0.2);
        requestBody.put("max_tokens", 2000);

        HttpHeaders headers = createOpenRouterHeaders();
        
        try {
            logger.info("Sending request to OpenRouter API with headers: {}", headers);
            ResponseEntity<Map<String, Object>> response = restTemplate.exchange(
                    OPENAI_API_URL,
                    HttpMethod.POST,
                    new HttpEntity<>(requestBody, headers),
                    new ParameterizedTypeReference<Map<String, Object>>() {
                    });

            return parseOpenAIResponse(response.getBody());
        } catch (Exception e) {
            logger.error("Conformity check error: {}", e.getMessage());
            return Map.of("error", e.getMessage());
        }
    }

    /**
     * Parses the OpenAI API response and extracts the conformity score and reasons.
     */
    public Map<String, Object> parseOpenAIResponse(Map<String, Object> response) {
        try {
            Object choicesObj = response.get("choices");
            if (!(choicesObj instanceof List)) {
                throw new RuntimeException("Format de réponse invalide pour 'choices'");
            }

            @SuppressWarnings("unchecked")
            List<Map<String, Object>> choices = (List<Map<String, Object>>) choicesObj;
            if (choices.isEmpty()) {
                throw new RuntimeException("Aucun choix disponible");
            }

            @SuppressWarnings("unchecked")
            Map<String, Object> firstChoice = choices.get(0);
            Object messageObj = firstChoice.get("message");
            if (!(messageObj instanceof Map)) {
                throw new RuntimeException("Format de message invalide");
            }

            @SuppressWarnings("unchecked")
            Map<String, Object> message = (Map<String, Object>) messageObj;
            Object contentObj = message.get("content");
            logger.info("LLM response content: {}", contentObj);  // ✅ Now shows actual response

            if (!(contentObj instanceof String)) {
                throw new RuntimeException("Contenu non textuel");
            }

            String content = (String) contentObj;

            // Extraction using regular expressions
            Pattern scorePattern = Pattern.compile("\\[SCORE:]\\s*(\\d+)"); // Explicit "SCORE:"
            Pattern consentPattern = Pattern.compile("\\[CONSENTEMENT:]\\s*(OUI|NON)", Pattern.CASE_INSENSITIVE);
            Pattern reasonsPattern = Pattern.compile("\\[RAISONS:\\]([\\s\\S]*?)(?=\\[|$)");

            Matcher scoreMatcher = scorePattern.matcher(content);
            Matcher consentMatcher = consentPattern.matcher(content);
            Matcher reasonsMatcher = reasonsPattern.matcher(content);

            int score = scoreMatcher.find() ? Integer.parseInt(scoreMatcher.group(1)) : 0;
            boolean consentement = consentMatcher.find() && consentMatcher.group(1).equalsIgnoreCase("OUI");

            String raisons = reasonsMatcher.find() ? reasonsMatcher.group(1).trim() : "Non spécifié";

            // Separate conformity points into two categories
            List<String> allReasons = Arrays.stream(raisons.split("\n"))
                    .map(String::trim)
                    .filter(s -> !s.isEmpty())
                    .collect(Collectors.toList());

            Map<String, List<String>> categorizedReasons = new HashMap<>();
            categorizedReasons.put("conform", new ArrayList<>());
            categorizedReasons.put("nonConform", new ArrayList<>());

            for (String reason : allReasons) {
                if (reason.toLowerCase().contains("non")) {
                    categorizedReasons.get("nonConform").add(reason);
                } else {
                    categorizedReasons.get("conform").add(reason);
                }
            }

            return Map.of(
                    "score_conformite", score,
                    "consentement_valide", consentement,
                    "points_conformite", categorizedReasons.get("conform"),
                    "points_non_conformite", categorizedReasons.get("nonConform"));
        } catch (Exception e) {
            logger.error("Erreur d'extraction: {}", e.getMessage());
            return Map.of("error", "Erreur de parsing de la réponse AI");
        }
    }    private HttpHeaders createOpenRouterHeaders() {
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        headers.set("Authorization", "Bearer " + OPENAI_API_KEY);
        
        // Required OpenRouter headers
        headers.set("HTTP-Referer", FRONTEND_URL); // The URL of your site
        headers.set("X-Title", "RGPD Analysis Tool"); // The name of your app
        headers.set("User-Agent", APP_USER_AGENT);
        
        return headers;
    }
   
}