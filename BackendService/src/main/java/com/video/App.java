package com.video;

import jakarta.annotation.PostConstruct;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.cache.annotation.EnableCaching;
import org.springframework.data.mongodb.repository.config.EnableMongoRepositories;

import java.util.TimeZone;

@SpringBootApplication
@EnableMongoRepositories
@EnableConfigurationProperties(AppConfig.class)
@EnableCaching
public class App {
    public static void main(String[] args) {
        SpringApplication.run(App.class, args);
    }

  @PostConstruct
  public void init(){
      TimeZone.setDefault(TimeZone.getTimeZone("UTC"));
  }
}


