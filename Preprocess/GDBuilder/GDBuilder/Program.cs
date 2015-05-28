using System;
using System.Collections.Generic;
using System.IO;

namespace GDBuilder
{
    class Program
    {
        private static readonly char[] SPACE_DELIM = new char[] { ' ' };

        static void Main(string[] args)
        {
            if (args.Length < 5)
            {
                PrintUsage();
                return;
            }

            //Read the corpora
            List<string> A = ReadInput(args[0]);
            Console.WriteLine(string.Format("READ A. {0} DOCUMENTS", A.Count));
            List<string> B = ReadInput(args[1]);
            Console.WriteLine(string.Format("READ B. {0} DOCUMENTS", B.Count));
            List<string> C = ReadInput(args[2]);
            Console.WriteLine(string.Format("READ C. {0} DOCUMENTS", C.Count));

            //Output the co-occurence dictionary, saving it to calc the distances
            Console.Write("BUILDING DICT...");
            Dictionary<string, Dictionary<string, int>> dict = OutputDict(A, args[4]);
            Console.WriteLine("DONE");

            //Output the distances
            Console.Write("BUILDING DICT...");
            OutputDistances(dict, args[3], B, C, A.Count);
            Console.WriteLine("DONE");
        }

        private static void PrintUsage()
        {
            Console.WriteLine("GDBuilder");
            Console.WriteLine("Turns 3 corpora, A, B & C into a word-word co-occurrence");
            Console.WriteLine("dictionary for A, and a set of distances");
            Console.WriteLine("between pairs of documents in B and C.");
            Console.WriteLine("B & C should be of the same length.");
            Console.WriteLine();
            Console.WriteLine("The co-occurrence dictionary is ouput in CSV format with header: ");
            Console.WriteLine("word1,word2,co");
            Console.WriteLine();
            Console.WriteLine("where word1 and word2 are two different words");
            Console.WriteLine("in alphabetical order and co is the");
            Console.WriteLine("number of documents in A where these");
            Console.WriteLine("words co-occurr. A blank word2 signifies co is the number of");
            Console.WriteLine("documents that contain word1.");
            Console.WriteLine();
            Console.WriteLine("The distances between documents in B & C are output");
            Console.WriteLine("as a text file, one distance per document pair.");
            Console.WriteLine();
            Console.WriteLine("Usage : GDBuilder a b c dist dict");
            Console.WriteLine();
            Console.WriteLine("where a, b and c are the paths to the corpora A, B & C");
            Console.WriteLine("respectively. Corpora should be text files with");
            Console.WriteLine("a single line of text per document(no newlines and\\or carriage returns).");
            Console.WriteLine("dist and dict are the paths to use for output of the distances and");
            Console.WriteLine("dictionary respectively");


        }

        private static double GoogleDistance(Dictionary<string, Dictionary<string, int>> dict,
            string word1, string word2, int size)
        {
            if (!dict.ContainsKey(word1) || !dict.ContainsKey(word2) || !dict[word1].ContainsKey(word2))
                return double.PositiveInfinity;

            double lw1 = Math.Log(dict[word1][string.Empty]);
            double lw2 = Math.Log(dict[word2][string.Empty]);
            double lw12 = Math.Log(dict[word1][word2]);
            return (Math.Max(lw1, lw2) - lw12) / (Math.Log(size) - Math.Min(lw1, lw2));
        }

        private static void OutputDistances(Dictionary<string, Dictionary<string, int>> dict,
            string path, List<string> corpus1, List<string> corpus2, int corpusSize)
        {
            using (StreamWriter sw = File.CreateText(path))
            {
                for (int i = 0; i < corpus1.Count; i++)
                {
                    string[] words1 = corpus1[i].Split(SPACE_DELIM, StringSplitOptions.RemoveEmptyEntries);
                    string[] words2 = corpus2[i].Split(SPACE_DELIM, StringSplitOptions.RemoveEmptyEntries);

                    double numD = 0;
                    double sumD = 0;
                    double phraseD = 0;
                    for (int w = 0; w < words1.Length; w++)
                    {
                        string w1 = words1[w];

                        for (int j = 0; j < words2.Length; j++)
                        {
                            string w2 = words2[j];
                            double d = GoogleDistance(dict, w1, w2, corpusSize);
                            if (!double.IsPositiveInfinity(d))
                            {
                                numD++;
                                sumD += d;
                            }
                        }
                    }
                    if (numD == 0)
                        phraseD = double.PositiveInfinity;
                    else if (sumD == 0)
                        phraseD = 0;
                    else
                        phraseD = sumD / numD;

                    if (double.IsPositiveInfinity(phraseD))
                        sw.WriteLine("Inf");
                    else
                        sw.WriteLine(phraseD);
                }
            }
        }

        private static Dictionary<string, Dictionary<string, int>> OutputDict(List<string> corpus, string path)
        {
            Dictionary<string, Dictionary<string, int>> result =
                new Dictionary<string, Dictionary<string, int>>(corpus.Count * 100);
            foreach (string phrase in corpus)
            {
                List<string> tokens = new List<string>(phrase.Split(SPACE_DELIM, StringSplitOptions.RemoveEmptyEntries));
                List<string> uniqueWords = new List<string>(tokens.Count);
                foreach (string word in tokens)
                {
                    if (!uniqueWords.Contains(word))
                        uniqueWords.Add(word);
                }

                //Count word ocurrences
                foreach (string word in uniqueWords)
                {
                    if (!result.ContainsKey(word))
                        result.Add(word, new Dictionary<string, int>());
                    if (!result[word].ContainsKey(string.Empty))
                        result[word].Add(string.Empty, 1);
                    else
                        result[word][string.Empty] = result[word][string.Empty] + 1;
                }

                //Count co-ocurrences
                for (int i = 0; i < uniqueWords.Count - 1; i++)
                {
                    string word1 = uniqueWords[i];
                    for (int j = i + 1; j < uniqueWords.Count; j++)
                    {
                        string word2 = uniqueWords[j];

                        if (!result[word1].ContainsKey(word2))
                            result[word1].Add(word2, 1);
                        else
                            result[word1][word2] = result[word1][word2] + 1;

                        if (!result[word2].ContainsKey(word1))
                            result[word2].Add(word1, 1);
                        else
                            result[word2][word1] = result[word2][word1] + 1;
                    }
                }
            }

            using (StreamWriter sw = File.CreateText(path))
            {
                Dictionary<string, bool> unique = new Dictionary<string, bool>(result.Keys.Count / 2);
                sw.WriteLine("word1,word2,co");
                List<string> wordPair = new List<string>() { string.Empty, string.Empty };
                foreach (KeyValuePair<string, Dictionary<string, int>> entry in result)
                {
                    foreach (KeyValuePair<string, int> word2 in entry.Value)
                    {
                        wordPair[0] = entry.Key;
                        wordPair[1] = word2.Key;
                        if (word2.Key != string.Empty)
                            wordPair.Sort();
                        string key = wordPair[0] + " " + wordPair[1];
                        if (!unique.ContainsKey(key))
                        {
                            unique.Add(key, true);
                            sw.Write(wordPair[0]);
                            sw.Write(',');
                            if (word2.Key != string.Empty)
                                sw.Write(wordPair[1]);
                            sw.Write(',');
                            sw.WriteLine(word2.Value);
                        }
                    }
                }
            }

            return result;
        }

        private static List<string> ReadInput(string path)
        {
            List<string> result = new List<string>();
            string line = null;
            using (StreamReader sr = File.OpenText(path))
            {
                while ((line = sr.ReadLine()) != null)
                    result.Add(line);
            }

            return result;
        }
    }
}
