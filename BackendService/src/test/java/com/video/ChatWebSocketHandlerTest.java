package com.video;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;
import org.redisson.api.RedissonClient;
import org.springframework.security.authentication.TestingAuthenticationToken;
import org.springframework.security.oauth2.core.user.DefaultOAuth2User;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.socket.TextMessage;
import org.springframework.web.socket.WebSocketSession;

import java.lang.reflect.Proxy;
import java.security.Principal;
import java.time.Instant;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicReference;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class ChatWebSocketHandlerTest {

    private final ObjectMapper objectMapper = new ObjectMapper();

    @Test
    void sessionListIncludesActiveSessionId() throws Exception {
        User user = user("user-1", "sub-1");
        ChatSession session = chatSession("session-1", "user-1");
        ChatUserState state = new ChatUserState();
        state.setUserId("user-1");
        state.setActiveSessionId("session-1");

        AtomicReference<TextMessage> sent = new AtomicReference<>();
        ChatWebSocketHandler handler = handler(
                userRepository("sub-1", user),
                sessionRepository(List.of(session), Optional.empty()),
                stateRepository(Optional.of(state)));

        handler.handleTextMessage(socket("sub-1", sent), new TextMessage("{\"type\":\"session.list\"}"));
        Map<?, ?> event = objectMapper.readValue(sent.get().getPayload(), Map.class);

        assertThat(event.get("type")).isEqualTo("session.listed");
        assertThat(event.get("activeSessionId")).isEqualTo("session-1");
    }

    @Test
    void sessionActivateRejectsSessionOwnedByAnotherUser() {
        User user = user("user-1", "sub-1");
        ChatSession otherSession = chatSession("session-2", "user-2");
        ChatWebSocketHandler handler = handler(
                userRepository("sub-1", user),
                sessionRepository(List.of(), Optional.of(otherSession)),
                stateRepository(Optional.empty()));

        assertThatThrownBy(() -> handler.handleTextMessage(
                socket("sub-1", new AtomicReference<>()),
                new TextMessage("{\"type\":\"session.activate\",\"sessionId\":\"session-2\"}")))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessage("Chat session not found");
    }

    private ChatWebSocketHandler handler(
            UserRepository userRepository,
            ChatSessionRepository chatSessionRepository,
            ChatUserStateRepository chatUserStateRepository) {
        return new ChatWebSocketHandler(
                objectMapper,
                userRepository,
                chatSessionRepository,
                proxy(ChatMessageRepository.class, null),
                chatUserStateRepository,
                new QdrantRestClient(new RestTemplate()),
                new EmbedderRestClient(new RestTemplate()),
                proxy(RedissonClient.class, null));
    }

    private UserRepository userRepository(String googleSub, User user) {
        return proxy(UserRepository.class, (method, args) -> {
            if ("findByGoogleSub".equals(method) && googleSub.equals(args[0])) {
                return Optional.of(user);
            }
            return defaultValue(Object.class);
        });
    }

    private ChatSessionRepository sessionRepository(List<ChatSession> sessions, Optional<ChatSession> byId) {
        return proxy(ChatSessionRepository.class, (method, args) -> switch (method) {
            case "findByUserIdOrderByUpdatedAtDesc" -> sessions;
            case "findById" -> byId;
            default -> defaultValue(Object.class);
        });
    }

    private ChatUserStateRepository stateRepository(Optional<ChatUserState> state) {
        return proxy(ChatUserStateRepository.class, (method, args) -> {
            if ("findById".equals(method)) return state;
            if ("save".equals(method)) return args[0];
            return defaultValue(methodReturnType(method));
        });
    }

    private WebSocketSession socket(String googleSub, AtomicReference<TextMessage> sent) {
        DefaultOAuth2User oauthUser = new DefaultOAuth2User(
                List.of(),
                Map.of("sub", googleSub),
                "sub");
        Principal principal = new TestingAuthenticationToken(oauthUser, null);
        return proxy(WebSocketSession.class, (method, args) -> switch (method) {
            case "getPrincipal" -> principal;
            case "isOpen" -> true;
            case "sendMessage" -> {
                sent.set((TextMessage) args[0]);
                yield null;
            }
            default -> defaultValue(methodReturnType(method));
        });
    }

    @SuppressWarnings("unchecked")
    private <T> T proxy(Class<T> type, Invocation invocation) {
        return (T) Proxy.newProxyInstance(
                type.getClassLoader(),
                new Class<?>[]{type},
                (target, method, args) -> {
                    if ("toString".equals(method.getName())) return type.getSimpleName() + " proxy";
                    if ("hashCode".equals(method.getName())) return System.identityHashCode(target);
                    if ("equals".equals(method.getName())) return target == args[0];
                    if (invocation != null) {
                        Object value = invocation.invoke(method.getName(), args == null ? new Object[0] : args);
                        if (value != null || method.getReturnType().equals(Void.TYPE)) return value;
                    }
                    return defaultValue(method.getReturnType());
                });
    }

    private Class<?> methodReturnType(String methodName) {
        return Object.class;
    }

    private Object defaultValue(Class<?> type) {
        if (type.equals(Boolean.TYPE)) return false;
        if (type.equals(Integer.TYPE)) return 0;
        if (type.equals(Long.TYPE)) return 0L;
        if (type.equals(Double.TYPE)) return 0.0;
        if (type.equals(Float.TYPE)) return 0.0f;
        if (type.equals(Void.TYPE)) return null;
        return null;
    }

    private User user(String userId, String googleSub) {
        User user = new User();
        user.setId(userId);
        user.setGoogleSub(googleSub);
        return user;
    }

    private ChatSession chatSession(String sessionId, String userId) {
        ChatSession session = new ChatSession();
        session.setId(sessionId);
        session.setUserId(userId);
        session.setTitle("Test chat");
        session.setCreatedAt(Instant.parse("2026-06-23T00:00:00Z"));
        session.setUpdatedAt(Instant.parse("2026-06-23T00:00:00Z"));
        return session;
    }

    private interface Invocation {
        Object invoke(String method, Object[] args);
    }
}
