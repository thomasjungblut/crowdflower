package de.jungblut.crowdflower;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.net.URL;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;

import au.com.bytecode.opencsv.CSVReader;
import de.jungblut.nlp.TokenizerUtils;

public class CrawlAmazon {

  private static final String ROOT_PATH = "../../";
  private static final String PROCESSED_PATH = "Processed/";

  static final int TOP_K_RESULTS = 10; // one request has max 15

  static int retrieved = 0;
  static int startId = 326680; // 10x max id from the trainingset

  public static void main(String[] args) throws IOException {

    System.setProperty("java.util.concurrent.ForkJoinPool.common.parallelism",
        "25");

    run(Paths.get(ROOT_PATH + PROCESSED_PATH + "train_scrubbed.csv"));
    System.out.println("Done!");
  }

  private static void run(Path path) throws IOException {

    final int queryIndex = 1;

    try (BufferedWriter writer = new BufferedWriter(new FileWriter(ROOT_PATH
        + PROCESSED_PATH + "amazon_train_data.csv"))) {

      writer
          .write("id,query,product_title,product_description,median_relevance,relevance_variance");
      writer.newLine();

      try (CSVReader reader = new CSVReader(new BufferedReader(new FileReader(
          path.toFile())), ',', '"', 1)) {
        List<String[]> all = reader.readAll();

        all.parallelStream()
            .forEach((line) -> {
              String query = line[queryIndex];
              List<String> productNames = retrieve(query);

              for (String productName : productNames) {
                // assume amazon always returns the highest relevance data with
                // zero
                // variance
                try {
                  synchronized (writer) {
                    writer.write(startId + ",\"" + query + "\",\""
                        + productName + "\",\"\",4,0");
                    writer.newLine();
                    startId++;
                  }
                } catch (Exception e) {
                }
              }

              synchronized (writer) {
                retrieved++;
                if (retrieved % 100 == 0) {
                  System.out.println("processed " + retrieved + " / "
                      + all.size());
                }
              }

            });
      }
    }
  }

  private static List<String> retrieve(String query) {
    ArrayList<String> list = new ArrayList<>();
    final int maxTries = 10;
    for (int tried = 1; tried <= maxTries; tried++) {
      try {
        URL url = new URL("http://www.amazon.com/s/?field-keywords="
            + query.replace(" ", "+"));

        Document doc = Jsoup.parse(url, 10_000);
        // xpath //*[@id="result_0"]/div/div/div/div[2]/div[1]/a/h2

        for (int i = 1; i <= TOP_K_RESULTS; i++) {
          String name = doc.select(
              "#result_" + i + " > div > div > div > div:eq(1) > div > a > h2")
              .text();
          name = TokenizerUtils.normalizeString(name.toLowerCase().trim());
          if (!name.isEmpty()) {
            list.add(name);
          }
        }
        break;
      } catch (Exception e) {
        if (tried != maxTries) {
          try {
            Thread.sleep((long) ((Math.pow(2, tried) - 1) * 1000));
          } catch (InterruptedException e1) {
            e1.printStackTrace();
          }
        } else {
          System.err.println("Exhausted max retries for query: " + query);
        }
      }
    }

    return list;
  }
}
