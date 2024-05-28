# Use an official OpenJDK runtime as a parent image
FROM openjdk:17

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the built JAR file into the container
COPY ./target/video-service-1.0.0.jar ./app.jar

# Expose port 8080 (or the port your application uses)
EXPOSE 8080

# Run the application
ENTRYPOINT ["java", "-Xmx2G", "-Xms1G", "-jar", "app.jar"]
