package com.video;

import jakarta.persistence.*;
import lombok.Data;

@Data
@Entity
@Table(name = "users")
public class User {

    @Id
    @GeneratedValue(strategy = GenerationType.UUID)
    private String id;

    @Column(unique = true, nullable = false)
    private String googleSub;

    @Column
    private String email;

    @Column
    private String name;

    @Column
    private String kmsKeyId;

    @Column
    private String kmsKeyArn;

    @Column(nullable = false)
    private String tenantId = "default";
}
