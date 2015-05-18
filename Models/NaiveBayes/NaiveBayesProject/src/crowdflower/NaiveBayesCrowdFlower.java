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

import org.tartarus.snowball.SnowballStemmer;
import org.tartarus.snowball.ext.englishStemmer;

import au.com.bytecode.opencsv.CSVReader;

import com.google.common.base.Preconditions;

import de.jungblut.classification.bayes.MultinomialNaiveBayes;
import de.jungblut.math.DoubleVector;
import de.jungblut.math.dense.DenseDoubleVector;
import de.jungblut.nlp.TokenizerUtils;
import de.jungblut.nlp.VectorizerUtils;

public class NaiveBayesCrowdFlower {

  private enum PrepMode {
    QUERY, TITLE, QUERY_AND_TITLE
  }

  private List<String> ids;
  private List<String[]> dicts;
  private DoubleVector[] trainVectors;
  private DoubleVector[] outcomes;

  private AveragingClassifier model;
  private List<String[]> documents;

  private void test(Path path, String testOutputPath) throws IOException {

    List<List<DoubleVector>> trainMatrix = new ArrayList<>();
    for (int i = 0; i < PrepMode.values().length; i++) {
      PrepMode mode = PrepMode.values()[i];
      prep(path, true, mode);
      vectorize(dicts.get(i));

      if (i == 0) {
        for (int n = 0; n < trainVectors.length; n++) {
          trainMatrix.add(new ArrayList<>());
        }
      }
      for (int n = 0; n < trainVectors.length; n++) {
        List<DoubleVector> list = trainMatrix.get(n);
        list.add(trainVectors[n]);
      }
    }

    try (BufferedWriter writer = new BufferedWriter(new FileWriter(
        testOutputPath + path.getFileName()))) {
      writer.write("\"id\",\"prediction\"\n");
      for (int i = 0; i < trainMatrix.size(); i++) {

        model.setFeatures(trainMatrix.get(i));
        DoubleVector predicted = model.predict(null);

        // DoubleVector predicted = model.predictProbability(trainVectors[i]);
        writer.write(ids.get(i) + "," + (predicted.maxIndex() + 1) + "\n");
      }
    }
  }

  private void train(Path path) throws IOException {

    dicts = new ArrayList<>();
    List<MultinomialNaiveBayes> subModels = new ArrayList<>();
    for (int i = 0; i < PrepMode.values().length; i++) {
      PrepMode mode = PrepMode.values()[i];
      prep(path, false, mode);
      String[] dict = null;
      if (mode == PrepMode.QUERY_AND_TITLE) {
        dict = VectorizerUtils.buildDictionary(documents.stream(), 0.2f, 3);
      } else {
        dict = VectorizerUtils.buildDictionary(documents.stream(), 1f, 0);
      }
      dicts.add(dict);
      vectorize(dict);

      MultinomialNaiveBayes m = new MultinomialNaiveBayes();
      m.train(trainVectors, outcomes);
      subModels.add(m);
    }

    model = new AveragingClassifier(subModels);

  }

  public void vectorize(String[] dict) {
    trainVectors = VectorizerUtils.wordFrequencyVectorize(documents.stream(),
        dict).toArray(i -> new DoubleVector[i]);
  }

  private void prep(Path path, boolean isTest, PrepMode mode)
      throws IOException {

    ids = new ArrayList<>();
    final SnowballStemmer stemmer = new englishStemmer();

    final DoubleVector ONE = new DenseDoubleVector(new double[] { 1d, 0, 0, 0 });
    final DoubleVector TWO = new DenseDoubleVector(new double[] { 0, 1d, 0, 0 });
    final DoubleVector THREE = new DenseDoubleVector(
        new double[] { 0, 0, 1d, 0 });
    final DoubleVector FOUR = new DenseDoubleVector(
        new double[] { 0, 0, 0, 1d });

    final int idIndex = 0;
    final int queryIndex = 1;
    final int titleIndex = 2;
    // final int descriptionIndex = 3;
    final int outcomeIndex = 4;

    List<DoubleVector> outcomesList = new ArrayList<>();
    documents = new ArrayList<>();
    try (CSVReader reader = new CSVReader(new BufferedReader(new FileReader(
        path.toFile())), ',', '"', 1)) {

      String[] line;
      while ((line = reader.readNext()) != null) {

        ids.add(line[idIndex]);
        List<String> tokenBag = new ArrayList<>();
        String fullText = "";

        switch (mode) {
          case QUERY:
            fullText = line[queryIndex];
            break;
          case TITLE:
            fullText = line[titleIndex];
            break;
          case QUERY_AND_TITLE:
            fullText = line[queryIndex] + " " + line[titleIndex];
            break;
        }

        fullText = TokenizerUtils.normalizeString(fullText);

        String[] tokens = TokenizerUtils.whiteSpaceTokenize(fullText);
        for (int i = 0; i < tokens.length; i++) {
          stemmer.setCurrent(tokens[i]);
          if (stemmer.stem()) {
            tokens[i] = stemmer.getCurrent();
          }
        }

        tokenBag.addAll(Arrays.asList(tokens));
        tokens = TokenizerUtils.addStartAndEndTags(tokens);
        tokenBag.addAll(Arrays.asList(TokenizerUtils.buildNGrams(tokens, 2)));
        tokenBag.addAll(Arrays.asList(TokenizerUtils.buildNGrams(tokens, 3)));
        tokenBag.addAll(Arrays.asList(TokenizerUtils.buildNGrams(tokens, 4)));

        documents.add(tokenBag.toArray(new String[tokenBag.size()]));
        if (!isTest) {
          switch (Integer.parseInt(line[outcomeIndex])) {
            case 1:
              outcomesList.add(ONE);
              break;
            case 2:
              outcomesList.add(TWO);
              break;
            case 3:
              outcomesList.add(THREE);
              break;
            case 4:
              outcomesList.add(FOUR);
              break;
            default:
              throw new IllegalArgumentException("unknown case");
          }
        }
      }

      if (!isTest) {
        Preconditions.checkArgument(documents.size() == outcomesList.size(),
            "docs/outcomes number of files didn't match");
      }

      outcomes = outcomesList.toArray(new DoubleVector[outcomesList.size()]);
    }
  }

  public static void main(String[] args) throws IOException {
    String basePath = "/Users/thomas.jungblut/git/crowdflower/ValidationFolds/";
    String testOutputPath = "/Users/thomas.jungblut/git/crowdflower/Models/NaiveBayes/";

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

      NaiveBayesCrowdFlower inst = new NaiveBayesCrowdFlower();

      inst.train(trainFile);

      inst.test(testFile, testOutputPath + "CvFoldsOutput/");

      System.out.print("\rFold " + (i + 1) + "/" + trainFiles.size());
    }

    System.out.println("\n\npredicting kaggle test set");

    NaiveBayesCrowdFlower inst = new NaiveBayesCrowdFlower();
    inst.train(Paths.get(basePath).resolveSibling("Raw/train.csv"));
    inst.test(Paths.get(basePath).resolveSibling("Raw/test.csv"),
        testOutputPath + "Submission/");

    System.out.println("\nDone.");
  }
}
