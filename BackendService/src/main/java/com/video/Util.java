package com.video;

import java.text.SimpleDateFormat;
import java.util.Date;

public class Util {

    private static String date_pattern = "yyyy/MM/dd/HH/mm/ss";
    private static String get_date_pattern = "yyyy/MM/dd/HH/mm/";

    public static String getS3Key(String fileName) {
        SimpleDateFormat dateFormat = new SimpleDateFormat(date_pattern);
        String strDate = dateFormat.format(new Date().getTime());
        return new StringBuilder()
            .append("videos/")
            .append(strDate)
            .append(getNameFromS3Key(fileName))
            .toString();
    }

    public static String getPrefixKeyForSearch(long startTime){
        SimpleDateFormat dateFormat = new SimpleDateFormat(get_date_pattern);
        String startDate = dateFormat.format(startTime);
        return new StringBuilder()
            .append("videos/")
            .append(startDate)
            .toString();
}

    public static String getLocalFilePath(String fileName){
        return new StringBuilder()
            .append("/tmp/")
            .append(fileName)
            .toString();
    }

    public static String getNameFromS3Key(String s3Key) {
        return s3Key.substring(s3Key.lastIndexOf('/') + 1);
    }
}
    
