package com.video;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.Id;
import jakarta.persistence.Table;
import lombok.Data;

import java.time.Instant;

@Data
@Entity
@Table(name = "chat_user_state")
public class ChatUserState {

    @Id
    private String userId;

    @Column
    private String activeSessionId;

    @Column(nullable = false)
    private Instant updatedAt = Instant.now();
}
