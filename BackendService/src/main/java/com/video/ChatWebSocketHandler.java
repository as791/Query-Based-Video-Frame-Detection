package com.video;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.security.core.Authentication;
import org.springframework.security.oauth2.core.user.OAuth2User;
import org.springframework.stereotype.Component;
import org.springframework.web.socket.CloseStatus;
import org.springframework.web.socket.TextMessage;
import org.springframework.web.socket.WebSocketSession;
import org.springframework.web.socket.handler.TextWebSocketHandler;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.time.Instant;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicLong;

@Component
@RequiredArgsConstructor
@Slf4j
public class ChatWebSocketHandler extends TextWebSocketHandler {

    private final ObjectMapper objectMapper;
    private final UserRepository userRepository;
    private final ChatSessionRepository chatSessionRepository;
    private final ChatMessageRepository chatMessageRepository;
    private final QdrantRestClient qdrantClient;
    private final EmbedderRestClient embedderClient;

    private final ExecutorService executor = Executors.newCachedThreadPool();

    @Value("${vlm.url:http://host.docker.internal:11434}")
    private String vlmUrl;

    @Value("${vlm.model:qwen2.5vl:7b}")
    private String vlmModel;

    @Override
    public void afterConnectionEstablished(WebSocketSession session) throws Exception {
        try {
            currentUser(session);
            sendEvent(session, "connected", Map.of("status", "ok"));
        } catch (Exception e) {
            session.close(CloseStatus.NOT_ACCEPTABLE.withReason("unauthorized"));
        }
    }

    @Override
    protected void handleTextMessage(WebSocketSession socket, TextMessage message) throws Exception {
        Map<String, Object> body = objectMapper.readValue(message.getPayload(), new TypeReference<>() {});
        String type = string(body.get("type"));
        User user = currentUser(socket);

        switch (type) {
            case "ping" -> sendEvent(socket, "pong", Map.of("ts", Instant.now().toString()));
            case "session.create" -> createSession(socket, user, body);
            case "session.list" -> listSessions(socket, user);
            case "messages.list" -> listMessages(socket, user, body);
            case "message.send" -> executor.submit(() -> sendMessage(socket, user, body));
            case "message.cancel" -> sendEvent(socket, "assistant.error", Map.of("message", "Cancel is not available in this local build yet."));
            default -> sendEvent(socket, "assistant.error", Map.of("message", "Unknown message type: " + type));
        }
    }

    private void createSession(WebSocketSession socket, User user, Map<String, Object> body) throws IOException {
        ChatSession session = new ChatSession();
        session.setUserId(user.getId());
        session.setVideoId(blankToNull(string(body.get("videoId"))));
        String title = string(body.get("title"));
        if (!title.isBlank()) session.setTitle(limit(title, 80));
        chatSessionRepository.save(session);
        sendEvent(socket, "session.created", Map.of("session", sessionDto(session)));
    }

    private void listSessions(WebSocketSession socket, User user) throws IOException {
        List<Map<String, Object>> sessions = chatSessionRepository.findByUserIdOrderByUpdatedAtDesc(user.getId())
                .stream()
                .map(this::sessionDto)
                .toList();
        sendEvent(socket, "session.listed", Map.of("sessions", sessions));
    }

    private void listMessages(WebSocketSession socket, User user, Map<String, Object> body) throws IOException {
        ChatSession session = requireSession(user, string(body.get("sessionId")));
        List<Map<String, Object>> messages = chatMessageRepository.findBySessionIdOrderBySeqAsc(session.getId())
                .stream()
                .map(this::messageDto)
                .toList();
        sendEvent(socket, "messages.listed", Map.of("sessionId", session.getId(), "messages", messages));
    }

    private void sendMessage(WebSocketSession socket, User user, Map<String, Object> body) {
        String clientMessageId = string(body.get("clientMessageId"));
        try {
            if (!clientMessageId.isBlank()) {
                Optional<ChatMessage> duplicate = chatMessageRepository.findByUserIdAndClientMessageIdAndRole(user.getId(), clientMessageId, "user");
                if (duplicate.isPresent()) {
                    sendEvent(socket, "assistant.error", Map.of(
                            "clientMessageId", clientMessageId,
                            "message", "Duplicate message ignored."));
                    return;
                }
            }

            String query = string(body.get("content"));
            if (query.isBlank()) {
                sendEvent(socket, "assistant.error", Map.of("clientMessageId", clientMessageId, "message", "Message content is required."));
                return;
            }

            ChatSession session = getOrCreateSession(user, body, query);
            long nextSeq = chatMessageRepository.countBySessionId(session.getId()) + 1;
            ChatMessage userMessage = new ChatMessage();
            userMessage.setSessionId(session.getId());
            userMessage.setUserId(user.getId());
            userMessage.setRole("user");
            userMessage.setContent(query);
            userMessage.setClientMessageId(blankToNull(clientMessageId));
            userMessage.setSeq(nextSeq);
            chatMessageRepository.save(userMessage);

            ChatMessage assistantMessage = new ChatMessage();
            assistantMessage.setSessionId(session.getId());
            assistantMessage.setUserId(user.getId());
            assistantMessage.setRole("assistant");
            assistantMessage.setStatus("streaming");
            assistantMessage.setSeq(nextSeq + 1);
            chatMessageRepository.save(assistantMessage);

            session.setUpdatedAt(Instant.now());
            chatSessionRepository.save(session);

            sendEvent(socket, "message.started", Map.of(
                    "session", sessionDto(session),
                    "userMessage", messageDto(userMessage),
                    "assistantMessage", messageDto(assistantMessage),
                    "clientMessageId", clientMessageId));

            List<Double> queryVector = embedderClient.embedText(query);
            List<Map<String, Object>> chunks = qdrantClient.searchChunks(queryVector, user.getId(), session.getVideoId(), 10);
            List<Map<String, Object>> sources = sourcesFromChunks(chunks);
            sendEvent(socket, "assistant.sources", Map.of(
                    "sessionId", session.getId(),
                    "messageId", assistantMessage.getId(),
                    "sources", sources));

            AtomicLong eventSeq = new AtomicLong(0);
            StringBuilder answer = new StringBuilder();
            boolean streamedFromVlm = streamFromVlm(socket, session, assistantMessage, clientMessageId, query, chunks, answer, eventSeq);
            if (!streamedFromVlm) {
                String fallback = fallbackAnswer(query, chunks);
                streamText(socket, session, assistantMessage, clientMessageId, fallback, answer, eventSeq);
            }

            assistantMessage.setContent(answer.toString());
            assistantMessage.setSourcesJson(objectMapper.writeValueAsString(sources));
            assistantMessage.setStatus("complete");
            chatMessageRepository.save(assistantMessage);
            sendEvent(socket, "assistant.completed", Map.of(
                    "sessionId", session.getId(),
                    "messageId", assistantMessage.getId(),
                    "clientMessageId", clientMessageId,
                    "seq", eventSeq.incrementAndGet()));
        } catch (Exception e) {
            log.error("WebSocket chat error", e);
            try {
                sendEvent(socket, "assistant.error", Map.of(
                        "clientMessageId", clientMessageId,
                        "message", e.getMessage() == null ? "Chat failed" : e.getMessage()));
            } catch (IOException ignored) {
            }
        }
    }

    private boolean streamFromVlm(
            WebSocketSession socket,
            ChatSession session,
            ChatMessage assistantMessage,
            String clientMessageId,
            String query,
            List<Map<String, Object>> chunks,
            StringBuilder answer,
            AtomicLong eventSeq) {
        try {
            String requestBody = objectMapper.writeValueAsString(Map.of(
                    "model", vlmModel,
                    "stream", true,
                    "messages", List.of(
                            Map.of("role", "system", "content", "You are a video assistant. Answer using the retrieved video context and cite timestamps when useful."),
                            Map.of("role", "user", "content", "Context:\n" + contextFromChunks(chunks) + "\n\nQuestion: " + query)
                    )
            ));

            HttpURLConnection conn = (HttpURLConnection) new URL(vlmUrl + "/v1/chat/completions").openConnection();
            conn.setConnectTimeout(3000);
            conn.setReadTimeout(120000);
            conn.setRequestMethod("POST");
            conn.setRequestProperty("Content-Type", "application/json");
            conn.setDoOutput(true);
            conn.getOutputStream().write(requestBody.getBytes(StandardCharsets.UTF_8));

            try (BufferedReader reader = new BufferedReader(new InputStreamReader(conn.getInputStream(), StandardCharsets.UTF_8))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    if (!line.startsWith("data:")) continue;
                    String data = line.substring(5).trim();
                    if ("[DONE]".equals(data)) break;
                    String delta = objectMapper.readTree(data).path("choices").path(0).path("delta").path("content").asText("");
                    if (!delta.isBlank()) {
                        sendDelta(socket, session.getId(), assistantMessage.getId(), clientMessageId, delta, eventSeq.incrementAndGet());
                        answer.append(delta);
                    }
                }
            }
            return answer.length() > 0;
        } catch (Exception e) {
            log.warn("VLM unavailable at {}; using retrieval-only WebSocket fallback: {}", vlmUrl, e.getMessage());
            return false;
        }
    }

    private void streamText(
            WebSocketSession socket,
            ChatSession session,
            ChatMessage assistantMessage,
            String clientMessageId,
            String text,
            StringBuilder answer,
            AtomicLong eventSeq) throws IOException {
        int index = 0;
        while (index < text.length()) {
            int end = Math.min(index + 96, text.length());
            String delta = text.substring(index, end);
            sendDelta(socket, session.getId(), assistantMessage.getId(), clientMessageId, delta, eventSeq.incrementAndGet());
            answer.append(delta);
            index = end;
        }
    }

    private void sendDelta(WebSocketSession socket, String sessionId, String messageId, String clientMessageId, String delta, long seq) throws IOException {
        sendEvent(socket, "assistant.delta", Map.of(
                "sessionId", sessionId,
                "messageId", messageId,
                "clientMessageId", clientMessageId,
                "delta", delta,
                "seq", seq));
    }

    private ChatSession getOrCreateSession(User user, Map<String, Object> body, String query) {
        String sessionId = string(body.get("sessionId"));
        if (!sessionId.isBlank()) {
            return requireSession(user, sessionId);
        }
        ChatSession session = new ChatSession();
        session.setUserId(user.getId());
        session.setVideoId(blankToNull(string(body.get("videoId"))));
        session.setTitle(limit(query, 60));
        return chatSessionRepository.save(session);
    }

    private ChatSession requireSession(User user, String sessionId) {
        ChatSession session = chatSessionRepository.findById(sessionId)
                .orElseThrow(() -> new IllegalArgumentException("Chat session not found"));
        if (!user.getId().equals(session.getUserId())) {
            throw new IllegalArgumentException("Chat session not found");
        }
        return session;
    }

    private User currentUser(WebSocketSession session) {
        if (session.getPrincipal() instanceof Authentication auth && auth.getPrincipal() instanceof OAuth2User oauth2User) {
            String sub = oauth2User.getAttribute("sub");
            return userRepository.findByGoogleSub(sub).orElseThrow(() -> new IllegalArgumentException("User not found"));
        }
        throw new IllegalArgumentException("Unauthorized");
    }

    private String contextFromChunks(List<Map<String, Object>> chunks) {
        StringBuilder context = new StringBuilder();
        for (Map<String, Object> chunk : chunks) {
            @SuppressWarnings("unchecked")
            Map<String, Object> payload = (Map<String, Object>) chunk.get("payload");
            if (payload == null) continue;
            String caption = String.valueOf(payload.getOrDefault("caption", "")).trim();
            long tStart = number(payload.get("t_start_ms"));
            long tEnd = number(payload.get("t_end_ms"));
            context.append("[").append(tStart / 1000).append("s-").append(tEnd / 1000).append("s]");
            if (!caption.isBlank()) context.append(" ").append(caption);
            context.append("\n");
        }
        return context.toString();
    }

    private List<Map<String, Object>> sourcesFromChunks(List<Map<String, Object>> chunks) {
        List<Map<String, Object>> sources = new ArrayList<>();
        for (Map<String, Object> chunk : chunks) {
            @SuppressWarnings("unchecked")
            Map<String, Object> payload = (Map<String, Object>) chunk.get("payload");
            if (payload == null) continue;
            sources.add(Map.of(
                    "videoId", String.valueOf(payload.getOrDefault("video_id", "")),
                    "chunkId", String.valueOf(payload.getOrDefault("chunk_id", "")),
                    "tStartMs", number(payload.get("t_start_ms")),
                    "tEndMs", number(payload.get("t_end_ms")),
                    "caption", String.valueOf(payload.getOrDefault("caption", ""))));
        }
        return sources;
    }

    private String fallbackAnswer(String query, List<Map<String, Object>> chunks) {
        if (chunks == null || chunks.isEmpty()) {
            return "I could not find indexed video context for: \"" + query + "\".";
        }

        StringBuilder answer = new StringBuilder();
        answer.append("I found relevant indexed video context, but the local VLM service is not running, so this is a retrieval-only answer.\n\n");
        answer.append("Most relevant segments:\n");
        for (Map<String, Object> source : sourcesFromChunks(chunks).stream().limit(5).toList()) {
            long start = (long) source.get("tStartMs");
            long end = (long) source.get("tEndMs");
            String videoId = String.valueOf(source.get("videoId"));
            String caption = String.valueOf(source.get("caption")).trim();
            answer.append("- ")
                    .append(start / 1000)
                    .append("s");
            if (end > start) answer.append("-").append(end / 1000).append("s");
            answer.append(" in video ").append(videoId.substring(0, Math.min(8, videoId.length())));
            if (!caption.isBlank()) answer.append(": ").append(caption);
            answer.append("\n");
        }
        answer.append("\nStart the `vlm` service to generate richer natural-language answers from these segments.");
        return answer.toString();
    }

    private Map<String, Object> sessionDto(ChatSession session) {
        Map<String, Object> dto = new LinkedHashMap<>();
        dto.put("id", session.getId());
        dto.put("title", session.getTitle());
        dto.put("videoId", session.getVideoId());
        dto.put("createdAt", session.getCreatedAt().toString());
        dto.put("updatedAt", session.getUpdatedAt().toString());
        return dto;
    }

    private Map<String, Object> messageDto(ChatMessage message) {
        Map<String, Object> dto = new LinkedHashMap<>();
        dto.put("id", message.getId());
        dto.put("sessionId", message.getSessionId());
        dto.put("role", message.getRole());
        dto.put("content", message.getContent());
        dto.put("status", message.getStatus());
        dto.put("sourcesJson", message.getSourcesJson());
        dto.put("clientMessageId", message.getClientMessageId());
        dto.put("seq", message.getSeq());
        dto.put("createdAt", message.getCreatedAt().toString());
        return dto;
    }

    private void sendEvent(WebSocketSession socket, String type, Map<String, ?> fields) throws IOException {
        if (!socket.isOpen()) return;
        Map<String, Object> event = new LinkedHashMap<>();
        event.put("type", type);
        event.putAll(fields);
        synchronized (socket) {
            if (socket.isOpen()) {
                socket.sendMessage(new TextMessage(objectMapper.writeValueAsString(event)));
            }
        }
    }

    private long number(Object value) {
        if (value instanceof Number number) return number.longValue();
        try {
            return value == null ? 0 : Long.parseLong(String.valueOf(value));
        } catch (NumberFormatException e) {
            return 0;
        }
    }

    private String string(Object value) {
        return value == null ? "" : String.valueOf(value);
    }

    private String blankToNull(String value) {
        return value == null || value.isBlank() ? null : value;
    }

    private String limit(String value, int max) {
        String normalized = value == null ? "" : value.trim();
        return normalized.length() <= max ? normalized : normalized.substring(0, max);
    }
}
