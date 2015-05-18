package crowdflower;

import java.util.List;

import com.google.common.base.Preconditions;

import de.jungblut.classification.AbstractClassifier;
import de.jungblut.math.DoubleVector;

public class AveragingClassifier extends AbstractClassifier {

  private List<? extends AbstractClassifier> subModels;
  private List<DoubleVector> features;

  public AveragingClassifier(List<? extends AbstractClassifier> subModels) {
    this.subModels = subModels;
  }

  public void setFeatures(List<DoubleVector> features) {
    Preconditions.checkArgument(features.size() == subModels.size(),
        "model and feature list sizes must match");
    this.features = features;
  }

  @Override
  public DoubleVector predict(DoubleVector unused) {

    DoubleVector sum = null;
    for (int i = 0; i < subModels.size(); i++) {
      AbstractClassifier classifier = subModels.get(i);
      DoubleVector pred = classifier.predictProbability(features.get(i));
      if (sum == null) {
        sum = pred;
      } else {
        sum = sum.add(pred);
      }
    }

    return sum.divide(sum.sum());
  }

  @Override
  public DoubleVector predictProbability(DoubleVector unused) {
    return predict(unused);
  }

}
