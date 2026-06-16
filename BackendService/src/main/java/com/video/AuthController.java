package com.video;

import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpStatus;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.security.oauth2.core.user.OAuth2User;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.server.ResponseStatusException;

@RestController
@CrossOrigin(originPatterns = {"${FRONTEND_ORIGIN:http://localhost:3000}", "http://localhost:*", "http://127.0.0.1:*"}, allowCredentials = "true")
@RequestMapping("v1/auth")
@RequiredArgsConstructor
public class AuthController {

    private final UserRepository userRepository;

    @GetMapping("/me")
    public User me(@AuthenticationPrincipal OAuth2User oAuth2User) {
        if (oAuth2User == null) {
            throw new ResponseStatusException(HttpStatus.UNAUTHORIZED);
        }
        String googleSub = oAuth2User.getAttribute("sub");
        return userRepository.findByGoogleSub(googleSub)
                .orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND));
    }
}
