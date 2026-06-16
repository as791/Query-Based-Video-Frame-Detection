package com.video;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.security.oauth2.core.user.OAuth2User;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.server.ResponseStatusException;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Executors;

@RestController
@CrossOrigin(originPatterns = {"${FRONTEND_ORIGIN:http://localhost:3000}", "http://localhost:*", "http://127.0.0.1:*"}, allowCredentials = "true")
@RequestMapping("v1/chat")
@RequiredArgsConstructor
@Slf4j
public class ChatController {

    private final QdrantRestClient qdrantClient;
    private final EmbedderRestClient embedderClient;
    private final UserRepository userRepository;

    @Value("${VLM_URL:http://host.docker.internal:11434}")
    private String vlmUrl;

    @Value("${VLM_MODEL:qwen2.5vl:7b}")
    private String vlmModel;

    @PostMapping(produces = MediaType.TEXT_EVENT_STREAM_VALUE)
    public SseEmitter chat(
            @RequestBody Map<String, Object> body,
            @AuthenticationPrincipal OAuth2User oAuth2User) {

        if (oAuth2User == null) throw new ResponseStatusException(HttpStatus.UNAUTHORIZED);

        User user = userRepository.findByGoogleSub(oAuth2User.getAttribute("sub"))
                .orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND));

        String query = (String) body.get("query");
        if (query == null || query.isBlank()) throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "query required");
        String videoId = (String) body.get("videoId");

        SseEmitter emitter = new SseEmitter(3 * 60 * 1000L);
        Executors.newSingleThreadExecutor().submit(() -> {
            try {
                List<Double> queryVector = embedderClient.embedText(query);
                List<Map<String, Object>> chunks = qdrantClient.searchChunks(queryVector, user.getId(), videoId, 10);

                StringBuilder context = new StringBuilder();
                for (Map<String, Object> chunk : chunks) {
                    @SuppressWarnings("unchecked")
                    Map<String, Object> payload = (Map<String, Object>) chunk.get("payload");
                    String caption = (String) payload.getOrDefault("caption", "");
                    long tStart = ((Number) payload.getOrDefault("t_start_ms", 0)).longValue();
                    if (!caption.isBlank()) context.append("[").append(tStart / 1000).append("s] ").append(caption).append("\n");
                }

                String systemPrompt = "video assistant";
                String userMessage = "Context:\n" + context + "\n\nQuestion: " + query;

                String requestBody = """
                        {"model":"%s","stream":true,"messages":[
                        {"role":"system","content":"%s"},
                        {"role":"user","content":"%s"}]}
                        """.formatted(vlmModel, systemPrompt, userMessage.replace("\"", "'"));

                try {
                    URL url = new URL(vlmUrl + "/v1/chat/completions");
                    HttpURLConnection conn = (HttpURLConnection) url.openConnection();
                    conn.setRequestMethod("POST");
                    conn.setRequestProperty("Content-Type", "application/json");
                    conn.setDoOutput(true);
                    conn.getOutputStream().write(requestBody.getBytes(StandardCharsets.UTF_8));

                    try (BufferedReader reader = new BufferedReader(new InputStreamReader(conn.getInputStream()))) {
                        String line;
                        while ((line = reader.readLine()) != null) {
                            if (line.startsWith("data: ")) emitter.send(SseEmitter.event().data(line.substring(6)));
                        }
                    }
                } catch (IOException e) {
                    log.warn("VLM unavailable at {}; using retrieval-only chat fallback: {}", vlmUrl, e.getMessage());
                    sendDelta(emitter, fallbackAnswer(query, chunks));
                }
                emitter.complete();
            } catch (Exception e) {
                log.error("Chat error", e);
                emitter.completeWithError(e);
            }
        });
        return emitter;
    }

    private String fallbackAnswer(String query, List<Map<String, Object>> chunks) {
        if (chunks == null || chunks.isEmpty()) {
            return "I could not find indexed video context for: \"" + query + "\".";
        }

        StringBuilder answer = new StringBuilder();
        answer.append("I found relevant indexed video context, but the local VLM service is not running, ");
        answer.append("so this is a retrieval-only answer.\n\n");
        answer.append("Most relevant segments:\n");

        int shown = 0;
        for (Map<String, Object> chunk : chunks) {
            if (shown >= 5) break;
            @SuppressWarnings("unchecked")
            Map<String, Object> payload = (Map<String, Object>) chunk.get("payload");
            long tStart = ((Number) payload.getOrDefault("t_start_ms", 0)).longValue();
            long tEnd = ((Number) payload.getOrDefault("t_end_ms", 0)).longValue();
            String videoId = String.valueOf(payload.getOrDefault("video_id", "unknown"));
            String caption = String.valueOf(payload.getOrDefault("caption", "")).trim();

            answer.append("- ")
                    .append(tStart / 1000)
                    .append("s");
            if (tEnd > tStart) {
                answer.append("-").append(tEnd / 1000).append("s");
            }
            answer.append(" in video ").append(videoId.substring(0, Math.min(8, videoId.length())));
            if (!caption.isBlank()) {
                answer.append(": ").append(caption);
            }
            answer.append("\n");
            shown++;
        }

        answer.append("\nStart the `vlm` service to generate richer natural-language answers from those segments.");
        return answer.toString();
    }

    private void sendDelta(SseEmitter emitter, String content) throws IOException {
        String escaped = content
                .replace("\\", "\\\\")
                .replace("\"", "\\\"")
                .replace("\n", "\\n")
                .replace("\r", "\\r");
        emitter.send(SseEmitter.event().data("{\"choices\":[{\"delta\":{\"content\":\"" + escaped + "\"}}]}"));
    }
}
