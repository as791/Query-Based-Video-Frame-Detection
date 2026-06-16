package com.video;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.security.oauth2.client.userinfo.DefaultOAuth2UserService;
import org.springframework.security.oauth2.client.oidc.userinfo.OidcUserRequest;
import org.springframework.security.oauth2.client.oidc.userinfo.OidcUserService;
import org.springframework.security.oauth2.client.userinfo.OAuth2UserRequest;
import org.springframework.security.oauth2.core.OAuth2AuthenticationException;
import org.springframework.security.oauth2.core.oidc.user.OidcUser;
import org.springframework.security.oauth2.core.user.OAuth2User;
import org.springframework.stereotype.Service;

@Service
@RequiredArgsConstructor
@Slf4j
public class CustomOAuth2UserService extends DefaultOAuth2UserService {

    private final UserRepository userRepository;
    private final KmsService kmsService;
    private final OidcUserService oidcUserService = new OidcUserService();

    @Override
    public OAuth2User loadUser(OAuth2UserRequest userRequest) throws OAuth2AuthenticationException {
        OAuth2User oAuth2User = super.loadUser(userRequest);
        provisionUser(oAuth2User.getAttribute("sub"), oAuth2User.getAttribute("email"), oAuth2User.getAttribute("name"));
        return oAuth2User;
    }

    public OidcUser loadOidcUser(OidcUserRequest userRequest) throws OAuth2AuthenticationException {
        OidcUser oidcUser = oidcUserService.loadUser(userRequest);
        provisionUser(oidcUser.getSubject(), oidcUser.getEmail(), oidcUser.getFullName());
        return oidcUser;
    }

    private void provisionUser(String googleSub, String email, String name) {
        userRepository.findByGoogleSub(googleSub).orElseGet(() -> {
            User newUser = new User();
            newUser.setGoogleSub(googleSub);
            newUser.setEmail(email);
            newUser.setName(name);
            newUser = userRepository.save(newUser);
            try {
                KmsService.KmsKeyResult kmsKey = kmsService.provisionUserKey(newUser.getId());
                newUser.setKmsKeyId(kmsKey.keyId());
                newUser.setKmsKeyArn(kmsKey.keyArn());
                newUser = userRepository.save(newUser);
                log.info("KMS key provisioned for new user {}: keyId={}", newUser.getId(), kmsKey.keyId());
            } catch (Exception e) {
                log.warn("KMS provisioning skipped for user {} (local dev or missing credentials): {}",
                        newUser.getId(), e.getMessage());
            }
            return newUser;
        });
    }
}
