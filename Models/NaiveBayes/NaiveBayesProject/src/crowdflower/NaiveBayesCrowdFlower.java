package crowdflower;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import au.com.bytecode.opencsv.CSVReader;

import com.google.common.base.Preconditions;

import de.jungblut.classification.bayes.MultinomialNaiveBayes;
import de.jungblut.math.DoubleVector;
import de.jungblut.math.dense.DenseDoubleVector;
import de.jungblut.math.tuple.Tuple;
import de.jungblut.nlp.TokenizerUtils;
import de.jungblut.nlp.VectorizerUtils;

public class NaiveBayesCrowdFlower {

  public static void main(String[] args) throws IOException {
    String basePath = "/Users/thomas.jungblut/git/crowdflower/ValidationFolds/";
    String testOutputPath = "/Users/thomas.jungblut/git/crowdflower/Models/NaiveBayes/CvFoldsOutput/";

    if (args.length > 0) {
      basePath = args[0];
      System.out.println("Using input path " + basePath);
      testOutputPath = args[1];
      System.out.println("Using test output path " + basePath);
    }

    List<Path> trainFiles = Files.list(Paths.get(basePath)).sorted()
        .filter(p -> p.toString().endsWith("train.csv"))
        .collect(Collectors.toList());

    List<Path> testFiles = Files.list(Paths.get(basePath)).sorted()
        .filter(p -> p.toString().endsWith("test.csv"))
        .collect(Collectors.toList());

    Preconditions.checkArgument(trainFiles.size() == testFiles.size(),
        "train/test number of files didn't match");

    for (int i = 0; i < trainFiles.size(); i++) {
      Path trainFile = trainFiles.get(i);
      Path testFile = testFiles.get(i);

      Preconditions.checkArgument(
          trainFile.getFileName().toString().split("-")[0].equals(testFile
              .getFileName().toString().split("-")[0]),
          "train/test fold files didn't match each other.");

      Tuple<String[], MultinomialNaiveBayes> model = train(trainFile);

      test(testFile, testOutputPath, model);

      System.out.print("\rFold " + (i + 1) + "/" + trainFiles.size());
    }
    System.out.println("Done.");
  }

  private static void test(Path path, String testOutputPath,
      Tuple<String[], MultinomialNaiveBayes> modelTuple) throws IOException {
    Tuple<List<String[]>, DoubleVector[]> prep = prep(path, true);
    String[] dict = modelTuple.getFirst();
    MultinomialNaiveBayes classifier = modelTuple.getSecond();
    List<String[]> documents = prep.getFirst();

    DoubleVector[] predVectors = VectorizerUtils.wordFrequencyVectorize(
        documents.stream(), dict).toArray(i -> new DoubleVector[i]);

    try (BufferedWriter writer = new BufferedWriter(new FileWriter(
        testOutputPath + path.getFileName()))) {
      writer.write("lineNumber,predictedClass\n");
      for (int i = 0; i < predVectors.length; i++) {
        DoubleVector predicted = classifier.predict(predVectors[i]);
        writer.write(i + "," + (predicted.maxIndex() + 1) + "\n");
      }
    }

  }

  private static Tuple<String[], MultinomialNaiveBayes> train(Path path)
      throws IOException {
    Tuple<List<String[]>, DoubleVector[]> prep = prep(path, false);
    List<String[]> documents = prep.getFirst();
    DoubleVector[] outcomes = prep.getSecond();

    String[] dict = VectorizerUtils
        .buildDictionary(documents.stream(), 0.5f, 3);
    DoubleVector[] trainVectors = VectorizerUtils.wordFrequencyVectorize(
        documents.stream(), dict).toArray(i -> new DoubleVector[i]);

    MultinomialNaiveBayes model = new MultinomialNaiveBayes();
    model.train(trainVectors, outcomes);

    return new Tuple<>(dict, model);
  }

  private static Tuple<List<String[]>, DoubleVector[]> prep(Path path,
      boolean isTest) throws IOException {

    final DoubleVector ONE = new DenseDoubleVector(new double[] { 1d, 0, 0, 0 });
    final DoubleVector TWO = new DenseDoubleVector(new double[] { 0, 1d, 0, 0 });
    final DoubleVector THREE = new DenseDoubleVector(
        new double[] { 0, 0, 1d, 0 });
    final DoubleVector FOUR = new DenseDoubleVector(
        new double[] { 0, 0, 0, 1d });

    final int queryIndex = 1;
    final int titleIndex = 2;
    final int descriptionIndex = 3;
    final int outcomeIndex = 4;

    List<DoubleVector> outcomes = new ArrayList<>();
    List<String[]> documents = new ArrayList<>();
    try (CSVReader reader = new CSVReader(new BufferedReader(new FileReader(
        path.toFile())), ',', '"', 1)) {

      String[] line;
      while ((line = reader.readNext()) != null) {

        List<String> tokenBag = new ArrayList<>();
        String fullText = line[queryIndex];
        fullText += " " + line[titleIndex];
        // FIXME don't use the description as it is too spammy
        // fullText += " " + Jsoup.parse(line[descriptionIndex]).text();

        fullText = TokenizerUtils.normalizeString(fullText);

        String[] tokens = TokenizerUtils.wordTokenize(fullText);
        tokenBag.addAll(Arrays.asList(tokens));
        tokenBag.addAll(Arrays.asList(TokenizerUtils.buildNGrams(tokens, 2)));

        documents.add(tokenBag.toArray(new String[tokenBag.size()]));
        if (!isTest) {
          switch (Integer.parseInt(line[outcomeIndex])) {
            case 1:
              outcomes.add(ONE);
              break;
            case 2:
              outcomes.add(TWO);
              break;
            case 3:
              outcomes.add(THREE);
              break;
            case 4:
              outcomes.add(FOUR);
              break;
            default:
              throw new IllegalArgumentException("unknown case");
          }
        }
      }

      if (!isTest) {
        Preconditions.checkArgument(documents.size() == outcomes.size(),
            "docs/outcomes number of files didn't match");
      }

      return new Tuple<>(documents, outcomes.toArray(new DoubleVector[outcomes
          .size()]));
    }
  }
}
