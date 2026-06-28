package com.video;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.redisson.api.RBucket;
import org.redisson.api.RLock;
import org.redisson.api.RTopic;
import org.redisson.api.RedissonClient;
import org.redisson.client.codec.StringCodec;
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
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.TimeUnit;

@Component
@RequiredArgsConstructor
@Slf4j
public class ChatWebSocketHandler extends TextWebSocketHandler {

    private final ObjectMapper objectMapper;
    private final UserRepository userRepository;
    private final ChatSessionRepository chatSessionRepository;
    private final ChatMessageRepository chatMessageRepository;
    private final ChatUserStateRepository chatUserStateRepository;
    private final QdrantRestClient qdrantClient;
    private final EmbedderRestClient embedderClient;
    private final RedissonClient redissonClient;

    private final ExecutorService executor = Executors.newCachedThreadPool();
    private final Map<String, Set<WebSocketSession>> socketsBySession = new ConcurrentHashMap<>();
    private final Map<WebSocketSession, Set<String>> sessionsBySocket = new ConcurrentHashMap<>();
    private final Map<String, Integer> topicListenerIds = new ConcurrentHashMap<>();

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
            case "session.activate" -> activateSession(socket, user, body);
            case "session.subscribe" -> subscribeSession(socket, user, body);
            case "messages.list" -> listMessages(socket, user, body);
            case "message.send" -> executor.submit(() -> sendMessage(socket, user, body));
            case "message.cancel" -> sendEvent(socket, "assistant.error", Map.of("message", "Cancel is not available in this local build yet."));
            default -> sendEvent(socket, "assistant.error", Map.of("message", "Unknown message type: " + type));
        }
    }

    @Override
    public void afterConnectionClosed(WebSocketSession session, CloseStatus status) {
        unsubscribeSocket(session);
    }

    private void createSession(WebSocketSession socket, User user, Map<String, Object> body) throws IOException {
        ChatSession session = new ChatSession();
        session.setUserId(user.getId());
        session.setVideoId(blankToNull(string(body.get("videoId"))));
        String title = string(body.get("title"));
        if (!title.isBlank()) session.setTitle(limit(title, 80));
        chatSessionRepository.save(session);
        activateUserSession(user, session.getId());
        sendEvent(socket, "session.created", Map.of("session", sessionDto(session)));
    }

    private void listSessions(WebSocketSession socket, User user) throws IOException {
        List<Map<String, Object>> sessions = chatSessionRepository.findByUserIdOrderByUpdatedAtDesc(user.getId())
                .stream()
                .map(this::sessionDto)
                .toList();
        sendEvent(socket, "session.listed", Map.of(
                "sessions", sessions,
                "activeSessionId", activeSessionId(user)));
    }

    private void activateSession(WebSocketSession socket, User user, Map<String, Object> body) throws IOException {
        ChatSession session = requireSession(user, string(body.get("sessionId")));
        activateUserSession(user, session.getId());
        sendEvent(socket, "session.activated", Map.of("session", sessionDto(session)));
    }

    private void subscribeSession(WebSocketSession socket, User user, Map<String, Object> body) throws IOException {
        ChatSession session = requireSession(user, string(body.get("sessionId")));
        activateUserSession(user, session.getId());
        subscribeSocket(socket, session.getId());
        sendEvent(socket, "session.subscribed", Map.of("sessionId", session.getId()));
    }

    private void listMessages(WebSocketSession socket, User user, Map<String, Object> body) throws IOException {
        ChatSession session = requireSession(user, string(body.get("sessionId")));
        activateUserSession(user, session.getId());
        List<Map<String, Object>> messages = messagesWithStreamingState(session.getId());
        sendEvent(socket, "messages.listed", Map.of("sessionId", session.getId(), "messages", messages));
    }

    private void sendMessage(WebSocketSession socket, User user, Map<String, Object> body) {
        String clientMessageId = string(body.get("clientMessageId"));
        RLock streamLock = null;
        try {
            if (!clientMessageId.isBlank()) {
                Optional<ChatMessage> duplicate = chatMessageRepository.findByUserIdAndClientMessageIdAndRole(user.getId(), clientMessageId, "user");
                if (duplicate.isPresent()) {
                    String sessionId = duplicate.get().getSessionId();
                    sendEvent(socket, "messages.listed", Map.of(
                            "sessionId", sessionId,
                            "messages", messagesWithStreamingState(sessionId)));
                    return;
                }
            }

            String query = string(body.get("content"));
            if (query.isBlank()) {
                sendEvent(socket, "assistant.error", Map.of("clientMessageId", clientMessageId, "message", "Message content is required."));
                return;
            }

            ChatSession session = getOrCreateSession(user, body, query);
            activateUserSession(user, session.getId());
            subscribeSocket(socket, session.getId());

            streamLock = redissonClient.getLock(streamLockKey(session.getId()));
            if (!streamLock.tryLock(0, 10, TimeUnit.MINUTES)) {
                publishSessionEvent(session.getId(), "assistant.error", Map.of(
                        "sessionId", session.getId(),
                        "clientMessageId", clientMessageId,
                        "message", "Assistant response already in progress for this chat."));
                return;
            }

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

            publishSessionEvent(session.getId(), "message.started", Map.of(
                    "sessionId", session.getId(),
                    "session", sessionDto(session),
                    "userMessage", messageDto(userMessage),
                    "assistantMessage", messageDto(assistantMessage),
                    "clientMessageId", clientMessageId));

            List<Double> queryVector = embedderClient.embedText(query);
            List<Map<String, Object>> chunks = qdrantClient.searchChunks(queryVector, user.getId(), session.getVideoId(), 10);
            List<Map<String, Object>> sources = sourcesFromChunks(chunks);
            publishSessionEvent(session.getId(), "assistant.sources", Map.of(
                    "sessionId", session.getId(),
                    "messageId", assistantMessage.getId(),
                    "clientMessageId", clientMessageId,
                    "sources", sources));

            AtomicLong eventSeq = new AtomicLong(0);
            AtomicInteger checkpointCounter = new AtomicInteger(0);
            StringBuilder answer = new StringBuilder();
            saveStreamingState(session.getId(), assistantMessage.getId(), answer.toString());
            boolean streamedFromVlm = streamFromVlm(session, assistantMessage, clientMessageId, query, chunks, answer, eventSeq, checkpointCounter);
            if (!streamedFromVlm) {
                String fallback = fallbackAnswer(query, chunks);
                streamText(session, assistantMessage, clientMessageId, fallback, answer, eventSeq, checkpointCounter);
            }

            assistantMessage.setContent(answer.toString());
            assistantMessage.setSourcesJson(objectMapper.writeValueAsString(sources));
            assistantMessage.setStatus("complete");
            chatMessageRepository.save(assistantMessage);
            clearStreamingState(session.getId());
            publishSessionEvent(session.getId(), "assistant.completed", Map.of(
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
        } finally {
            if (streamLock != null && streamLock.isHeldByCurrentThread()) {
                streamLock.unlock();
            }
        }
    }

    private boolean streamFromVlm(
            ChatSession session,
            ChatMessage assistantMessage,
            String clientMessageId,
            String query,
            List<Map<String, Object>> chunks,
            StringBuilder answer,
            AtomicLong eventSeq,
            AtomicInteger checkpointCounter) {
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
                        publishDelta(session, assistantMessage, clientMessageId, delta, answer, eventSeq, checkpointCounter);
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
            ChatSession session,
            ChatMessage assistantMessage,
            String clientMessageId,
            String text,
            StringBuilder answer,
            AtomicLong eventSeq,
            AtomicInteger checkpointCounter) throws IOException {
        int index = 0;
        while (index < text.length()) {
            int end = Math.min(index + 96, text.length());
            String delta = text.substring(index, end);
            publishDelta(session, assistantMessage, clientMessageId, delta, answer, eventSeq, checkpointCounter);
            index = end;
        }
    }

    private void publishDelta(
            ChatSession session,
            ChatMessage assistantMessage,
            String clientMessageId,
            String delta,
            StringBuilder answer,
            AtomicLong eventSeq,
            AtomicInteger checkpointCounter) throws IOException {
        answer.append(delta);
        assistantMessage.setContent(answer.toString());
        assistantMessage.setStatus("streaming");
        saveStreamingState(session.getId(), assistantMessage.getId(), answer.toString());
        if (checkpointCounter.incrementAndGet() % 5 == 0) {
            chatMessageRepository.save(assistantMessage);
        }
        publishSessionEvent(session.getId(), "assistant.delta", Map.of(
                "sessionId", session.getId(),
                "messageId", assistantMessage.getId(),
                "clientMessageId", clientMessageId,
                "delta", delta,
                "seq", eventSeq.incrementAndGet()));
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
        ChatSession saved = chatSessionRepository.save(session);
        activateUserSession(user, saved.getId());
        return saved;
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

    private void activateUserSession(User user, String sessionId) {
        ChatUserState state = chatUserStateRepository.findById(user.getId()).orElseGet(() -> {
            ChatUserState next = new ChatUserState();
            next.setUserId(user.getId());
            return next;
        });
        state.setActiveSessionId(sessionId);
        state.setUpdatedAt(Instant.now());
        chatUserStateRepository.save(state);
    }

    private String activeSessionId(User user) {
        return chatUserStateRepository.findById(user.getId())
                .map(ChatUserState::getActiveSessionId)
                .orElse("");
    }

    private List<Map<String, Object>> messagesWithStreamingState(String sessionId) {
        List<Map<String, Object>> messages = chatMessageRepository.findBySessionIdOrderBySeqAsc(sessionId)
                .stream()
                .map(this::messageDto)
                .collect(java.util.stream.Collectors.toCollection(ArrayList::new));
        StreamingState streamingState = streamingState(sessionId);
        if (streamingState == null) return messages;
        for (Map<String, Object> message : messages) {
            if (streamingState.messageId().equals(string(message.get("id")))) {
                message.put("content", streamingState.content());
                message.put("status", "streaming");
                return messages;
            }
        }
        return messages;
    }

    private void subscribeSocket(WebSocketSession socket, String sessionId) {
        sessionsBySocket.computeIfAbsent(socket, key -> ConcurrentHashMap.newKeySet()).add(sessionId);
        socketsBySession.computeIfAbsent(sessionId, key -> ConcurrentHashMap.newKeySet()).add(socket);
        topicListenerIds.computeIfAbsent(sessionId, this::registerTopicListener);
    }

    private int registerTopicListener(String sessionId) {
        RTopic topic = redissonClient.getTopic(topicName(sessionId), StringCodec.INSTANCE);
        return topic.addListener(String.class, (channel, message) -> fanOutSessionEvent(sessionId, message));
    }

    private void unsubscribeSocket(WebSocketSession socket) {
        Set<String> sessionIds = sessionsBySocket.remove(socket);
        if (sessionIds == null) return;
        for (String sessionId : sessionIds) {
            Set<WebSocketSession> sockets = socketsBySession.get(sessionId);
            if (sockets == null) continue;
            sockets.remove(socket);
            if (sockets.isEmpty()) {
                socketsBySession.remove(sessionId);
                Integer listenerId = topicListenerIds.remove(sessionId);
                if (listenerId != null) {
                    redissonClient.getTopic(topicName(sessionId), StringCodec.INSTANCE).removeListener(listenerId);
                }
            }
        }
    }

    private void fanOutSessionEvent(String sessionId, String message) {
        Set<WebSocketSession> sockets = socketsBySession.getOrDefault(sessionId, Collections.emptySet());
        for (WebSocketSession socket : sockets) {
            try {
                if (socket.isOpen()) {
                    synchronized (socket) {
                        if (socket.isOpen()) {
                            socket.sendMessage(new TextMessage(message));
                        }
                    }
                }
            } catch (Exception e) {
                log.debug("Failed to fan out chat event for session {}", sessionId, e);
            }
        }
    }

    private void publishSessionEvent(String sessionId, String type, Map<String, ?> fields) throws IOException {
        Map<String, Object> event = new LinkedHashMap<>();
        event.put("type", type);
        event.putAll(fields);
        redissonClient.getTopic(topicName(sessionId), StringCodec.INSTANCE)
                .publish(objectMapper.writeValueAsString(event));
    }

    private void saveStreamingState(String sessionId, String messageId, String content) throws IOException {
        StreamingState state = new StreamingState(messageId, content == null ? "" : content);
        RBucket<String> bucket = redissonClient.getBucket(streamStateKey(sessionId), StringCodec.INSTANCE);
        bucket.set(objectMapper.writeValueAsString(state), 10, TimeUnit.MINUTES);
    }

    private StreamingState streamingState(String sessionId) {
        try {
            String raw = redissonClient.<String>getBucket(streamStateKey(sessionId), StringCodec.INSTANCE).get();
            if (raw == null || raw.isBlank()) return null;
            return objectMapper.readValue(raw, StreamingState.class);
        } catch (Exception e) {
            log.debug("Could not read streaming state for session {}", sessionId, e);
            return null;
        }
    }

    private void clearStreamingState(String sessionId) {
        redissonClient.getBucket(streamStateKey(sessionId), StringCodec.INSTANCE).delete();
    }

    private String topicName(String sessionId) {
        return "chat:session:" + sessionId + ":events";
    }

    private String streamStateKey(String sessionId) {
        return "chat:session:" + sessionId + ":stream-state";
    }

    private String streamLockKey(String sessionId) {
        return "chat:session:" + sessionId + ":stream-lock";
    }

    private record StreamingState(String messageId, String content) {}

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
