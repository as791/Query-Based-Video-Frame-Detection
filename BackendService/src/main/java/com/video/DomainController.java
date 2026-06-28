package com.video;

import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpStatus;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.security.oauth2.core.user.OAuth2User;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.server.ResponseStatusException;

import java.util.Map;

@RestController
@CrossOrigin(originPatterns = {"${FRONTEND_ORIGIN:http://localhost:3000}", "http://localhost:*", "http://127.0.0.1:*"}, allowCredentials = "true")
@RequestMapping("v1/domains")
@RequiredArgsConstructor
public class DomainController {

    private final DomainModelService domainModelService;
    private final FewShotLearningService fewShotLearningService;
    private final UserRepository userRepository;

    @GetMapping
    public DomainModelService.DomainList list(@AuthenticationPrincipal OAuth2User oAuth2User) {
        User user = requireUser(oAuth2User);
        return domainModelService.list(user.getId());
    }

    @PostMapping
    public DomainModelService.DomainState create(
            @RequestBody Map<String, Object> body,
            @AuthenticationPrincipal OAuth2User oAuth2User) {
        User user = requireUser(oAuth2User);
        return domainModelService.createDomain(
                user.getId(),
                String.valueOf(body.getOrDefault("name", "")),
                String.valueOf(body.getOrDefault("description", "")));
    }

    @GetMapping("/{domainId}/model")
    public DomainModelService.DomainState model(
            @PathVariable String domainId,
            @AuthenticationPrincipal OAuth2User oAuth2User) {
        User user = requireUser(oAuth2User);
        return domainModelService.state(user.getId(), domainId);
    }

    @PostMapping("/{domainId}/labels")
    public DomainModelService.DomainLabel addLabel(
            @PathVariable String domainId,
            @RequestBody Map<String, Object> body,
            @AuthenticationPrincipal OAuth2User oAuth2User) {
        User user = requireUser(oAuth2User);
        try {
            DomainModelService.DomainLabel label = domainModelService.addLabel(
                    user.getId(),
                    domainId,
                    String.valueOf(body.getOrDefault("label", "")),
                    String.valueOf(body.getOrDefault("description", "")));
            fewShotLearningService.addLabel(
                    user.getId(),
                    domainId,
                    label.label(),
                    label.description());
            return label;
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
