package com.video;

import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.security.core.authority.SimpleGrantedAuthority;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.security.oauth2.client.authentication.OAuth2AuthenticationToken;
import org.springframework.security.oauth2.core.user.DefaultOAuth2User;
import org.springframework.stereotype.Component;
import org.springframework.web.filter.OncePerRequestFilter;

import java.io.IOException;
import java.util.List;
import java.util.Map;

@Component
public class BenchmarkAuthFilter extends OncePerRequestFilter {

    @Value("${benchmark.auth.enabled:false}")
    private boolean enabled;

    @Value("${benchmark.auth.google-sub:}")
    private String allowedGoogleSub;

    @Override
    protected void doFilterInternal(HttpServletRequest request, HttpServletResponse response, FilterChain filterChain)
            throws ServletException, IOException {
        if (enabled && SecurityContextHolder.getContext().getAuthentication() == null) {
            String googleSub = request.getHeader("X-Benchmark-Google-Sub");
            if (googleSub != null && !googleSub.isBlank() && googleSub.equals(allowedGoogleSub)) {
                var authorities = List.of(new SimpleGrantedAuthority("ROLE_USER"));
                var principal = new DefaultOAuth2User(authorities, Map.of(
                        "sub", googleSub,
                        "email", request.getHeader("X-Benchmark-Email") == null ? "benchmark@local" : request.getHeader("X-Benchmark-Email"),
                        "name", "Benchmark User"), "sub");
                var authentication = new OAuth2AuthenticationToken(principal, authorities, "benchmark");
                SecurityContextHolder.getContext().setAuthentication(authentication);
            }
        }
        filterChain.doFilter(request, response);
    }
}
