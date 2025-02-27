package de.jungblut.crowdflower;

import gnu.trove.list.array.TDoubleArrayList;
import gnu.trove.set.hash.TCharHashSet;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.HashSet;

import org.tartarus.snowball.SnowballStemmer;
import org.tartarus.snowball.ext.englishStemmer;

import au.com.bytecode.opencsv.CSVReader;

import com.google.common.collect.HashMultiset;
import com.google.common.collect.Multiset.Entry;

import de.jungblut.datastructure.ArrayJoiner;
import de.jungblut.nlp.TokenizerUtils;

public class CreateFeatures {

  private static final String ROOT_PATH = "../../../";
  private static final String PROCESSED_PATH = "Processed/";

  public static void main(String[] args) throws IOException {
    run(Paths.get(ROOT_PATH + PROCESSED_PATH + "train_scrubbed.csv"), false);
    run(Paths.get(ROOT_PATH + PROCESSED_PATH + "test_scrubbed.csv"), true);
    System.out.println("Done!");
  }

  private static final String HEADER = "id"
      + ",queryentropy,titleentropy" //
      + ",btitleinquery,longestprefix,longestsuffix,longestQueryPrefixPerc,longestQuerySuffixPerc,longestTitlePrefixPerc,longestTitleSuffixPerc"
      + ",numtitledigits,numquerydigits,numdigitdiff,bindigiteq,charjaccardsim,charcossim"
      + ",charquerycount3,chartitlecount3,charngcount3,charngjacc3,charngcos3" //
      + ",charquerycount4,chartitlecount4,charngcount4,charngjacc4,charngcos4" //
      + ",charquerycount5,chartitlecount5,charngcount5,charngjacc5,charngcos5" //
      + ",charquerycount6,chartitlecount6,charngcount6,charngjacc6,charngcos6"//
      + ",charquerycount7,chartitlecount7,charngcount7,charngjacc7,charngcos7"//
      + ",wordquerycount1,wordtitlecount1,wordngcount1,leftngcount1,wordngjacc1,wordngcos1" //
      + ",wordquerycount2,wordtitlecount2,wordngcount2,leftngcount2,wordngjacc2,wordngcos2" //
      + ",wordquerycount3,wordtitlecount3,wordngcount3,leftngcount3,wordngjacc3,wordngcos3" //
      + ",wordquerycount4,wordtitlecount4,wordngcount4,leftngcount4,wordngjacc4,wordngcos4" //
      + ",wordquerycount5,wordtitlecount5,wordngcount5,leftngcount5,wordngjacc5,wordngcos5";

  private static double[] computeFeatures(String query, String title) {
    TDoubleArrayList list = new TDoubleArrayList(70);

    list.add(entropy(query));
    list.add(entropy(title));

    list.add(title.contains(query) ? 1 : 0);

    // some general features for strings
    int longestPrefix = processLongestPrefix(query, title).length();
    list.add(longestPrefix);
    int longestSuffix = processLongestSuffix(query, title).length();
    list.add(longestSuffix);

    if (query.length() == 0) {
      list.add(0);
      list.add(0);
    } else {
      list.add(longestPrefix / (double) query.length());
      list.add(longestSuffix / (double) query.length());
    }

    if (title.length() == 0) {
      list.add(0);
      list.add(0);
    } else {
      list.add(longestPrefix / (double) title.length());
      list.add(longestSuffix / (double) title.length());
    }

    int titleDigit = countDigit(title);
    int queryDigit = countDigit(query);
    list.add(titleDigit);
    list.add(queryDigit);
    list.add(Math.abs(titleDigit - queryDigit));
    list.add(titleDigit == queryDigit ? 1 : 0);

    list.add(measureJaccardSimilarity(query.toCharArray(), title.toCharArray()));
    list.add(measureCosineSimilarity(query.toCharArray(), title.toCharArray()));

    // char ngrams
    for (int i = 3; i < 8; i++) {
      String[] queryQGram = TokenizerUtils.qGramTokenize(query, i);
      list.add(queryQGram.length);
      String[] titleQGram = TokenizerUtils.qGramTokenize(title, i);
      list.add(titleQGram.length);
      int match = countMatches(queryQGram, titleQGram);
      list.add(match);
      double jacc = measureJaccardSimilarity(queryQGram, titleQGram);
      list.add(jacc);
      double cos = measureCosineSimilarity(queryQGram, titleQGram);
      list.add(cos);
    }

    // word ngrams
    for (int i = 1; i < 6; i++) {
      String[] queryNgram = wordTokenize(query);
      list.add(queryNgram.length);
      String[] titleNgram = wordTokenize(title);
      list.add(titleNgram.length);

      list.add(countMatches(queryNgram, titleNgram));
      list.add(getNGramLeftOver(queryNgram, titleNgram));

      double jacc = measureJaccardSimilarity(queryNgram, titleNgram);
      list.add(jacc);
      double cos = measureCosineSimilarity(queryNgram, titleNgram);
      list.add(cos);
    }

    return list.toArray();
  }

  // shannon entropy of the chars in a string
  private static double entropy(String s) {
    HashMultiset<Character> set = HashMultiset.create();
    for (char c : s.toCharArray()) {
      set.add(c);
    }

    double result = 0.0;
    int len = s.length();

    for (Entry<Character> item : set.entrySet()) {
      double frequency = (double) item.getCount() / len;
      result -= frequency * (Math.log(frequency) / Math.log(2));
    }

    return result;
  }

  private static int getNGramLeftOver(String[] queryTokens, String[] titleTokens) {
    HashSet<String> intersection = new HashSet<>();

    for (int i = 0; i < queryTokens.length; i++) {
      intersection.add(queryTokens[i]);
    }

    for (int i = 0; i < titleTokens.length; i++) {
      intersection.remove(titleTokens[i]);
    }

    return intersection.size();
  }

  private static int countDigit(String s) {
    int x = 0;
    char[] charArray = s.toCharArray();
    for (int i = 0; i < charArray.length; i++) {
      if (Character.isDigit(charArray[i])) {
        x++;
      }
    }
    return x;
  }

  private static String[] wordTokenize(String s) {
    final SnowballStemmer stemmer = new englishStemmer();

    String trimmed = s.toLowerCase().trim();
    String[] wordTokenize = TokenizerUtils.wordTokenize(trimmed);

    for (int i = 0; i < wordTokenize.length; i++) {
      stemmer.setCurrent(wordTokenize[i]);
      if (stemmer.stem()) {
        wordTokenize[i] = stemmer.getCurrent();
      }
    }

    return wordTokenize;
  }

  private static int countMatches(String[] queryTokens, String[] titleTokens) {
    HashMultiset<String> set = HashMultiset.create();

    set.addAll(Arrays.asList(queryTokens));
    set.addAll(Arrays.asList(titleTokens));

    return set.elementSet().size();
  }

  private static String processLongestSuffix(String title, String name) {
    String shorter = title.length() > name.length() ? name : title;
    String lengthier = shorter == title ? name : title;
    int diff = lengthier.length() - shorter.length();
    int i = lengthier.length() - 1;
    for (; i >= 0 && i - diff >= 0; i--) {
      if (lengthier.charAt(i) != shorter.charAt(i - diff)) {
        break;
      }
    }
    if (i >= 0) {
      return new String(lengthier.substring(i + 1, lengthier.length()));
    }
    return "";
  }

  private static String processLongestPrefix(String title, String name) {

    int i = 0;
    for (; i < Math.min(title.length(), name.length()); i++) {
      if (name.charAt(i) != title.charAt(i)) {
        break;
      }
    }

    return new String(name.substring(0, i));
  }

  private static double measureJaccardSimilarity(char[] left, char[] right) {
    if (right == null || left == null) {
      return 0;
    }

    HashSet<Character> intersection = new HashSet<>();
    HashSet<Character> union = new HashSet<>(right.length + left.length);

    for (int i = 0; i < right.length; i++) {
      intersection.add(right[i]);
      union.add(right[i]);
    }

    for (int i = 0; i < left.length; i++) {
      union.add(left[i]);
      intersection.remove(left[i]);
    }

    return ((double) intersection.size()) / (union.size());
  }

  private static double measureJaccardSimilarity(String[] left, String[] right) {
    if (right == null || left == null) {
      return 0;
    }

    HashSet<String> intersection = new HashSet<>();
    HashSet<String> union = new HashSet<>(right.length + left.length);

    for (int i = 0; i < right.length; i++) {
      intersection.add(right[i]);
      union.add(right[i]);
    }

    for (int i = 0; i < left.length; i++) {
      union.add(left[i]);
      intersection.remove(left[i]);
    }

    return ((double) intersection.size()) / (union.size());
  }

  private static double measureCosineSimilarity(char[] left, char[] right) {
    if (right == null || left == null) {
      return 0;
    }

    TCharHashSet union = new TCharHashSet(left);
    int termsInLeftString = union.size();
    TCharHashSet secondStringTokens = new TCharHashSet(right);
    int termsInRightString = secondStringTokens.size();

    union.addAll(secondStringTokens);
    int commonTerms = (termsInLeftString + termsInRightString) - union.size();

    return ((double) commonTerms)
        / (Math.pow(termsInLeftString, 0.5) * Math.pow(termsInRightString, 0.5));
  }

  private static double measureCosineSimilarity(String[] left, String[] right) {
    if (right == null || left == null) {
      return 0;
    }

    HashSet<String> union = new HashSet<String>(Arrays.asList(left));
    int termsInLeftString = union.size();
    HashSet<String> secondStringTokens = new HashSet<String>(
        Arrays.asList(right));
    int termsInRightString = secondStringTokens.size();

    union.addAll(secondStringTokens);
    int commonTerms = (termsInLeftString + termsInRightString) - union.size();

    return ((double) commonTerms)
        / (Math.pow(termsInLeftString, 0.5) * Math.pow(termsInRightString, 0.5));
  }

  private static void run(Path path, boolean isTest) throws IOException {
    final int idIndex = 0;
    final int queryIndex = 1;
    final int titleIndex = 2;
    // final int descriptionIndex = 3;
    final int outcomeIndex = 4;

    try (BufferedWriter writer = new BufferedWriter(new FileWriter(ROOT_PATH
        + PROCESSED_PATH + "distances_features" + (isTest ? "_test" : "")
        + ".csv"))) {

      writer.write(HEADER);
      if (!isTest) {
        writer.write(",outcome");
      }
      writer.newLine();

      try (CSVReader reader = new CSVReader(new BufferedReader(new FileReader(
          path.toFile())), ',', '"', 1)) {

        String[] line;
        while ((line = reader.readNext()) != null) {

          double[] features = computeFeatures(line[queryIndex],
              line[titleIndex]);

          writer
              .write(line[idIndex] + "," + ArrayJoiner.on(',').join(features));
          if (!isTest) {
            writer.write("," + line[outcomeIndex]);
          }
          writer.newLine();
        }

      }
    }
  }
}
