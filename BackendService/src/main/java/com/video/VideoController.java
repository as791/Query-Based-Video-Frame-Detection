package com.video;

import lombok.RequiredArgsConstructor;

import java.util.List;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

@RestController
@CrossOrigin
@RequestMapping("v1/video")
public class VideoController {

    @Autowired
    private VideoService videoService;


    @PostMapping("/uploadVideo")
    public VideoMetadata uploadVideo(@RequestBody MultipartFile file) {
       return videoService.uploadRecordedVideo(file);
    }


    @GetMapping("/search")
    public SearchOutputLinks searchOnVideo(@RequestParam("query") String query, @RequestParam("startTime") long startTime, @RequestParam("endTime") long endTime) {
      return videoService.searchOnVideo(new SeachRequestDTO(query, startTime, endTime));
    }
}
