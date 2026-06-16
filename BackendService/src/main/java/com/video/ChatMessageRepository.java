package com.video;

import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;
import java.util.Optional;

public interface ChatMessageRepository extends JpaRepository<ChatMessage, String> {
    List<ChatMessage> findBySessionIdOrderBySeqAsc(String sessionId);
    Optional<ChatMessage> findByUserIdAndClientMessageIdAndRole(String userId, String clientMessageId, String role);
    long countBySessionId(String sessionId);
}
