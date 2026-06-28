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
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.server.ResponseStatusException;

import java.util.List;
import java.util.Map;

@RestController
@CrossOrigin(originPatterns = {"${FRONTEND_ORIGIN:http://localhost:3000}", "http://localhost:*", "http://127.0.0.1:*"}, allowCredentials = "true")
@RequestMapping("v1/action-labels")
@RequiredArgsConstructor
public class ActionTaxonomyController {

    private final ActionTaxonomyService actionTaxonomyService;

    @GetMapping
    public Map<String, List<ActionTaxonomyService.ActionLabel>> labels(@AuthenticationPrincipal OAuth2User oAuth2User) {
        if (oAuth2User == null) throw new ResponseStatusException(HttpStatus.UNAUTHORIZED);
        return Map.of("labels", actionTaxonomyService.labels());
    }

    @PostMapping
    public ActionTaxonomyService.ActionLabel add(
            @RequestBody Map<String, Object> body,
            @AuthenticationPrincipal OAuth2User oAuth2User) {
        if (oAuth2User == null) throw new ResponseStatusException(HttpStatus.UNAUTHORIZED);
        try {
            return actionTaxonomyService.add(
                    String.valueOf(body.getOrDefault("label", "")),
                    String.valueOf(body.getOrDefault("description", "")));
        } catch (IllegalArgumentException e) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, e.getMessage(), e);
        }
    }
}
