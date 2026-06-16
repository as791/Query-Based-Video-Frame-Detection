package com.video;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.Id;
import jakarta.persistence.Table;
import lombok.Data;

import java.time.Instant;
import java.util.UUID;

@Data
@Entity
@Table(name = "chat_messages")
public class ChatMessage {

    @Id
    private String id = UUID.randomUUID().toString();

    @Column(nullable = false)
    private String sessionId;

    @Column(nullable = false)
    private String userId;

    @Column(nullable = false)
    private String role;

    @Column(nullable = false, columnDefinition = "text")
    private String content = "";

    @Column(nullable = false)
    private String status = "complete";

    @Column(columnDefinition = "text")
    private String sourcesJson;

    @Column
    private String clientMessageId;

    @Column(nullable = false)
    private Long seq = 0L;

    @Column(nullable = false)
    private Instant createdAt = Instant.now();
}
