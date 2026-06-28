package com.video;

import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpStatus;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.security.oauth2.core.user.OAuth2User;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.server.ResponseStatusException;

import java.util.Map;

@RestController
@CrossOrigin(originPatterns = {"${FRONTEND_ORIGIN:http://localhost:3000}", "http://localhost:*", "http://127.0.0.1:*"}, allowCredentials = "true")
@RequestMapping("v1/few-shot")
@RequiredArgsConstructor
public class FewShotLearningController {

    private final FewShotLearningService fewShotLearningService;
    private final UserRepository userRepository;

    @GetMapping
    public FewShotLearningService.FewShotState state(
            @RequestParam(required = false, defaultValue = "general") String domainId,
            @AuthenticationPrincipal OAuth2User oAuth2User) {
        User user = requireUser(oAuth2User);
        return fewShotLearningService.state(user.getId(), domainId);
    }

    @PostMapping("/labels")
    public FewShotLearningService.FewShotLabel addLabel(
            @RequestBody Map<String, Object> body,
            @RequestParam(required = false, defaultValue = "general") String domainId,
            @AuthenticationPrincipal OAuth2User oAuth2User) {
        User user = requireUser(oAuth2User);
        try {
            return fewShotLearningService.addLabel(
                    user.getId(),
                    domainId,
                    String.valueOf(body.getOrDefault("label", "")),
                    String.valueOf(body.getOrDefault("description", "")));
        } catch (IllegalArgumentException e) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, e.getMessage(), e);
        }
    }

    private User requireUser(OAuth2User oAuth2User) {
        if (oAuth2User == null) throw new ResponseStatusException(HttpStatus.UNAUTHORIZED);
        return userRepository.findByGoogleSub(oAuth2User.getAttribute("sub"))
                .orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND));
    }
}
