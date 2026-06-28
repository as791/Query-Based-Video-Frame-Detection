package com.video;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.http.MediaType;
import org.springframework.security.test.web.servlet.request.SecurityMockMvcRequestPostProcessors;
import org.springframework.test.context.ActiveProfiles;
import org.springframework.test.web.servlet.MockMvc;

import static org.springframework.security.test.web.servlet.request.SecurityMockMvcRequestPostProcessors.oauth2Login;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.*;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.*;

@SpringBootTest
@AutoConfigureMockMvc
@ActiveProfiles("test")
class ApiIntegrationTest {

    @Autowired
    MockMvc mvc;

    @Test
    void unauthenticated_me_returns_401() throws Exception {
        mvc.perform(get("/v1/auth/me"))
                .andExpect(status().isUnauthorized());
    }

    @Test
    void authenticated_me_returns_user_or_404() throws Exception {
        // oauth2Login() injects a fake OAuth2 principal — no real Google call needed
        mvc.perform(get("/v1/auth/me")
                        .with(oauth2Login()
                                .attributes(a -> {
                                    a.put("sub",   "test-google-sub-123");
                                    a.put("email", "test@example.com");
                                    a.put("name",  "Test User");
                                })))
                .andExpect(result -> expectStatusOneOf(result.getResponse().getStatus(), 200, 404)); // 200 if user exists, 404 if not yet registered
    }

    @Test
    void unauthenticated_presign_returns_401() throws Exception {
        mvc.perform(post("/v1/video/presignUpload"))
                .andExpect(status().isUnauthorized());
    }

    @Test
    void unauthenticated_search_returns_401() throws Exception {
        mvc.perform(post("/v1/search")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("{\"query\":\"a dog\"}"))
                .andExpect(status().isUnauthorized());
    }

    @Test
    void search_with_blank_query_returns_400() throws Exception {
        mvc.perform(post("/v1/search")
                        .with(oauth2Login()
                                .attributes(a -> {
                                    a.put("sub",   "test-google-sub-123");
                                    a.put("email", "test@example.com");
                                }))
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("{\"query\":\"\"}"))
                .andExpect(result -> expectStatusOneOf(result.getResponse().getStatus(), 400, 404)); // 400 for blank query, 404 if user missing
    }

    @Test
    void presign_authenticated_unknown_user_returns_404() throws Exception {
        mvc.perform(post("/v1/video/presignUpload")
                        .with(oauth2Login()
                                .attributes(a -> {
                                    a.put("sub",   "unknown-sub-xyz");
                                    a.put("email", "unknown@example.com");
                                })))
                .andExpect(status().isNotFound());
    }

    private static void expectStatusOneOf(int actual, int first, int second) {
        if (actual != first && actual != second) {
            throw new AssertionError("expected " + first + " or " + second + ", got " + actual);
        }
    }
}
