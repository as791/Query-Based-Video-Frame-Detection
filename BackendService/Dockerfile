# ── Build stage ──────────────────────────────────────────────────────────────
FROM maven:3.9-eclipse-temurin-17 AS builder

WORKDIR /build

# Copy pom first to cache dependency downloads
COPY pom.xml ./
RUN mvn -B dependency:go-offline

# Copy sources and build
COPY src ./src
RUN mvn -B clean package -Dmaven.test.skip=true

# ── Runtime stage ────────────────────────────────────────────────────────────
FROM eclipse-temurin:17-jre

WORKDIR /usr/src/app

COPY --from=builder /build/target/video-service-1.0.0.jar ./app.jar

EXPOSE 8080

ENTRYPOINT ["java", "-Xmx2G", "-Xms1G", "-jar", "app.jar"]
