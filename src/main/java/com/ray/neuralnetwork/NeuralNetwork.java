package com.ray.neuralnetwork;

import java.util.ArrayList;
import java.util.Collections;


public class NeuralNetwork implements java.io.Serializable {
	int[] layers;
	ArrayList<Matrix> weights;
	ArrayList<Matrix> biases;
	
	public NeuralNetwork(int[] lyers) {
		layers = lyers;
		int len = layers.length;
		weights = new ArrayList<Matrix> ();
		biases = new ArrayList<Matrix> ();
		
		for (int i = 1; i < len; ++i) {
			weights.add(new Matrix(layers[i], layers[i - 1], true));
			biases.add(new Matrix(layers[i], 1, true));
		}
		
	}
	
	public void printNetworkWeights() {
		for (int i = 0; i < weights.size(); ++i) {
			System.out.print("***** <Weights> *****\n");
			weights.get(i).printMatrix();
			System.out.print("***** </Weights> *****\n\n");
		}
	}
	
	public void printNetworkBiases() {
		for (int i = 0; i < biases.size(); ++i) {
			System.out.print("***** <Biases> *****\n");
			biases.get(i).printMatrix();
			System.out.print("***** </Biases> *****\n\n");
		}
	}
	
	public void update(WBPair wb, double rate) {
		int len = layers.length;
		for (int n = 0; n < len - 1; ++n) {
			weights.set(n, weights.get(n).minus(wb.getWeightsAt(n).mult(rate)));
			biases.set(n, biases.get(n).minus(wb.getBiasesAt(n).mult(rate)));
		}
	}
	
	// Sigmoid *************************************************
	
	public Matrix feedForward(Matrix input) {
		if ((input.getRowNum() != layers[0]) || (input.getColNum() != 1)) {
			System.out.print("Invalid input vector to feedforward.");
			return null;
		}
		
		Matrix container = weights.get(0).mult(input).add(biases.get(0));
		container = container.sigmoid();
		for (int i = 1; i < weights.size(); ++i) {
			container = weights.get(i).mult(container).add(biases.get(i));
			container = container.sigmoid();
		}
		return container;
	}
	
	public Matrix feedForwardRawAt(Matrix input, int index) {
		if (index == 0) {
			return input;
		}
		if (index == 1) {
			return weights.get(0).mult(input).add(biases.get(0));
		}
		Matrix result = weights.get(0).mult(input).add(biases.get(0));
		for (int i = 1; i < index; ++i) {
			result = result.sigmoid();
			result = weights.get(i).mult(result).add(biases.get(i));
		}
		return result;
	}
	
	public Matrix feedForwardOutputAt(Matrix input, int index) {
		if (index == 0) {
			return input;
		}
		return this.feedForwardRawAt(input, index).sigmoid();
	}
	
	// Sigmoid + Quadratic Cost function ************************* (SQC)
	
	public WBPair backPropogateSQC(Matrix input, Matrix expect) {
		int len = layers.length;
		ArrayList<Matrix> errors = new ArrayList<Matrix>();
		ArrayList<Matrix> weights_delta = new ArrayList<Matrix>();
		ArrayList<Matrix> biases_delta = new ArrayList<Matrix>();
		
		ArrayList<Matrix> rawOutput = new ArrayList<Matrix>();
		ArrayList<Matrix> sigmoidOutput = new ArrayList<Matrix>();
		
		Matrix walker = new Matrix(input.getRowNum(), input.getColNum(), false);
		for (int r = 0; r < walker.getRowNum(); ++r) {
			for (int c = 0; c < walker.getColNum(); ++c) {
				walker.setEntry(r, c, input.getEntry(r, c));
			}
		}
		rawOutput.add(walker);
		sigmoidOutput.add(walker);
		for (int i = 0; i < len - 1; ++i) {
			walker = weights.get(i).mult(walker).add(biases.get(i));
			rawOutput.add(walker);
			walker = walker.sigmoid();
			sigmoidOutput.add(walker);
			
		}
		
		
		
		//Matrix output = feedForward(input);
		Matrix output = sigmoidOutput.get(len - 1);
		
		Matrix gradientC = output.minus(expect);
		
		//Matrix errorL = gradientC.hadamard(this.feedForwardRawAt(input, len - 1).sigmoidPrime());
		Matrix errorL = gradientC.hadamard(rawOutput.get(len - 1).sigmoidPrime());
		
		errors.add(errorL);
		
		/*
		for (int j = len - 2; j > 0; --j) {
			errors.add(weights.get(j).transpose().mult(errors.get(len - 2 - j)).hadamard(feedForwardRawAt(input, j).sigmoidPrime()));
		}
		*/
		for (int j = len - 2; j > 0; --j) {
			errors.add(weights.get(j).transpose().mult(errors.get(len - 2 - j)).hadamard(rawOutput.get(j).sigmoidPrime()));
		}
		Collections.reverse(errors);
		
		biases_delta = errors;
		
		/*
		for (int m = 0; m < len - 1; ++m) {
			weights_delta.add(errors.get(m).mult(this.feedForwardOutputAt(input, m).transpose()));
		}
		*/
		
		for (int m = 0; m < len - 1; ++m) {
			weights_delta.add(errors.get(m).mult(sigmoidOutput.get(m).transpose()));
		}
		
		return new WBPair(weights_delta, biases_delta);
	}
	
	public double costQuadratic(Matrix input, Matrix expect) {
		Matrix output = feedForward(input);
		Matrix diff = expect.minus(output);
		double result = 0;
		for (int i = 0; i < diff.getRowNum(); ++i) {
			result += Math.pow(diff.getEntry(i, 0), 2);
		}
		return result / 2;
	}
	
	public void trainSQC(ArrayList<TrainingData> data, double rate, int epoch) {
		int size = data.size();
		for (int i = 0; i < epoch; ++i) {
			double c = 0;
			for (int j = 0; j < size; ++j) {
				WBPair wb = backPropogateSQC(data.get(j).input, data.get(j).expect);
				c += this.costQuadratic(data.get(j).input, data.get(j).expect);
				update(wb, rate);
			
			}
			//System.out.print("\nEpoch " + i + " completed. The average cost is " + (c / data.size()) + "\n");
			System.out.print("\nEpoch " + i + " completed. The average cost is " + (c / size) + "\n");
		}
		System.out.print("\nTraining completed.\n");
	}
	
	public void trainStochasticSQC(ArrayList<TrainingData> data, ArrayList<TrainingData> testData, int batchSize, double rate, int epoch) {
		for (int z = 0; z < epoch; ++z) {
			Collections.shuffle(data);
			int size = data.size();
			int batchNum = size / batchSize;
			int lastBatchSize = size % batchSize;
			double c = 0;
			if (lastBatchSize == 0) {
				
				for (int i = 0; i < batchNum; ++i) {
					ArrayList<Matrix> w = new ArrayList<Matrix>();
					ArrayList<Matrix> b = new ArrayList<Matrix>();
					
					for (int p = 1; p < layers.length; ++p) {
						w.add(new Matrix(layers[p], layers[p - 1], false));
						b.add(new Matrix(layers[p], 1, false));
					}
					WBPair wb = new WBPair(w, b); 
					for (int j = 0; j < batchSize; ++j) {
						wb = wb.add(this.backPropogateSQC(data.get(i * batchSize + j).input, data.get(i * batchSize + j).expect));
						c += this.costQuadratic(data.get(i * batchSize + j).input, data.get(i * batchSize + j).expect);
					}
					wb = wb.divide(batchSize);
					this.update(wb, rate);
				}
				
			} else {
				
				for (int i = 0; i < batchNum - 1; ++i) {
					ArrayList<Matrix> w = new ArrayList<Matrix>();
					ArrayList<Matrix> b = new ArrayList<Matrix>();
					
					for (int p = 1; p < layers.length; ++p) {
						w.add(new Matrix(layers[p], layers[p - 1], false));
						b.add(new Matrix(layers[p], 1, false));
					}
					WBPair wb = new WBPair(w, b); 
					for (int j = 0; j < batchSize; ++j) {
						wb = wb.add(this.backPropogateSQC(data.get(i * batchSize + j).input, data.get(i * batchSize + j).expect));
						c += this.costQuadratic(data.get(i * batchSize + j).input, data.get(i * batchSize + j).expect);
					}
					wb = wb.divide(batchSize);
					this.update(wb, rate);
				}
				
				for (int p = 0; p < lastBatchSize; ++p) {
					ArrayList<Matrix> w = new ArrayList<Matrix>();
					ArrayList<Matrix> b = new ArrayList<Matrix>();
					
					for (int i = 1; i < layers.length; ++i) {
						w.add(new Matrix(layers[i], layers[i - 1], false));
						b.add(new Matrix(layers[i], 1, false));
					}
					WBPair wb = new WBPair(w, b); 
					
					wb = wb.add(this.backPropogateSQC(data.get((batchNum - 1) * batchSize + p).input, data.get((batchNum - 1) * batchSize + p).expect));
					c += this.costQuadratic(data.get((batchNum - 1) * batchSize + p).input, data.get((batchNum - 1) * batchSize + p).expect);
					
					wb = wb.divide(lastBatchSize);
					this.update(wb, rate);
				}
				
			}
			System.out.print("\nEpoch " + z + " completed. The average cost is " + (c / size) + "\n");
			this.test(testData);
		}
		System.out.print("\nTraining completed.\n");
	}
	
	public void trainStochasticSQC(ArrayList<TrainingData> data, int batchSize, double rate, int epoch) {
		for (int z = 0; z < epoch; ++z) {
			Collections.shuffle(data);
			int size = data.size();
			int batchNum = size / batchSize;
			int lastBatchSize = size % batchSize;
			double c = 0;
			if (lastBatchSize == 0) {
				
				for (int i = 0; i < batchNum; ++i) {
					ArrayList<Matrix> w = new ArrayList<Matrix>();
					ArrayList<Matrix> b = new ArrayList<Matrix>();
					
					for (int p = 1; p < layers.length; ++p) {
						w.add(new Matrix(layers[p], layers[p - 1], false));
						b.add(new Matrix(layers[p], 1, false));
					}
					WBPair wb = new WBPair(w, b); 
					for (int j = 0; j < batchSize; ++j) {
						wb = wb.add(this.backPropogateSQC(data.get(i * batchSize + j).input, data.get(i * batchSize + j).expect));
						c += this.costQuadratic(data.get(i * batchSize + j).input, data.get(i * batchSize + j).expect);
					}
					wb = wb.divide(batchSize);
					this.update(wb, rate);
				}
				
			} else {
				
				for (int i = 0; i < batchNum - 1; ++i) {
					ArrayList<Matrix> w = new ArrayList<Matrix>();
					ArrayList<Matrix> b = new ArrayList<Matrix>();
					
					for (int p = 1; p < layers.length; ++p) {
						w.add(new Matrix(layers[p], layers[p - 1], false));
						b.add(new Matrix(layers[p], 1, false));
					}
					WBPair wb = new WBPair(w, b); 
					for (int j = 0; j < batchSize; ++j) {
						wb = wb.add(this.backPropogateSQC(data.get(i * batchSize + j).input, data.get(i * batchSize + j).expect));
						c += this.costQuadratic(data.get(i * batchSize + j).input, data.get(i * batchSize + j).expect);
					}
					wb = wb.divide(batchSize);
					this.update(wb, rate);
				}
				
				for (int p = 0; p < lastBatchSize; ++p) {
					ArrayList<Matrix> w = new ArrayList<Matrix>();
					ArrayList<Matrix> b = new ArrayList<Matrix>();
					
					for (int i = 1; i < layers.length; ++i) {
						w.add(new Matrix(layers[i], layers[i - 1], false));
						b.add(new Matrix(layers[i], 1, false));
					}
					WBPair wb = new WBPair(w, b); 
					
					wb = wb.add(this.backPropogateSQC(data.get((batchNum - 1) * batchSize + p).input, data.get((batchNum - 1) * batchSize + p).expect));
					c += this.costQuadratic(data.get((batchNum - 1) * batchSize + p).input, data.get((batchNum - 1) * batchSize + p).expect);
					
					wb = wb.divide(lastBatchSize);
					this.update(wb, rate);
				}
				
			}
			System.out.print("\nEpoch " + z + " completed. The average cost is " + (c / size) + "\n");
		}
		System.out.print("\nTraining completed.\n");
	}
	
	// End of Sigmoid + Quadratic Cost (SQC) ************************
	
	// Sigmoid + Cross-Entropy (SCE) ******************************
	
	public WBPair backPropogateSCE(Matrix input, Matrix expect) {
		ArrayList<Matrix> errors = new ArrayList<Matrix>();
		ArrayList<Matrix> weights_delta = new ArrayList<Matrix>();
		ArrayList<Matrix> biases_delta = new ArrayList<Matrix>();
		
		int len = layers.length;
		
		Matrix output = feedForward(input);
		
		Matrix errorL = output.minus(expect);
		
		errors.add(errorL);
		for (int j = len - 2; j > 0; --j) {
			errors.add(weights.get(j).transpose().mult(errors.get(len - 2 - j)).hadamard(feedForwardRawAt(input, j).sigmoidPrime()));
		}
		Collections.reverse(errors);
		
		biases_delta = errors;
		
		for (int m = 0; m < len - 1; ++m) {
			weights_delta.add(errors.get(m).mult(this.feedForwardOutputAt(input, m).transpose()));
		}
		
		return new WBPair(weights_delta, biases_delta);
	}
	
	public double costEntropy(Matrix input, Matrix expect) {
		Matrix output = feedForward(input);
		
		double result = 0;
		for (int i = 0; i < expect.getRowNum(); ++i) {
			result += -(expect.getEntry(i, 0) * Math.log(output.getEntry(i, 0)) + (1 - expect.getEntry(i, 0)) * Math.log(1 - output.getEntry(i, 0)));
		}
		return result;
	}
	
	public void trainSCE(ArrayList<TrainingData> data, double rate, int epoch) {
		int size = data.size();
		for (int i = 0; i < epoch; ++i) {
			double c = 0;
			for (int j = 0; j < data.size(); ++j) {
				WBPair wb = backPropogateSCE(data.get(j).input, data.get(j).expect);
				c += this.costEntropy(data.get(j).input, data.get(j).expect);
				update(wb, rate);
			}
			//System.out.print("\nEpoch " + i + " completed. The average cost is " + (c / data.size()) + "\n");
			System.out.print("\nEpoch " + i + " completed. The average cost is " + (c / size) + "\n");
		}
		System.out.print("\nTraining completed.\n");
	}
	
	public void trainStochasticSCE(ArrayList<TrainingData> data, ArrayList<TrainingData> testData, int batchSize, double rate, int epoch) {
		for (int z = 0; z < epoch; ++z) {
			Collections.shuffle(data);
			int size = data.size();
			int batchNum = size / batchSize;
			int lastBatchSize = size % batchSize;
			double c = 0;
			if (lastBatchSize == 0) {
				
				for (int i = 0; i < batchNum; ++i) {
					ArrayList<Matrix> w = new ArrayList<Matrix>();
					ArrayList<Matrix> b = new ArrayList<Matrix>();
					
					for (int p = 1; p < layers.length; ++p) {
						w.add(new Matrix(layers[p], layers[p - 1], false));
						b.add(new Matrix(layers[p], 1, false));
					}
					WBPair wb = new WBPair(w, b); 
					for (int j = 0; j < batchSize; ++j) {
						wb = wb.add(this.backPropogateSCE(data.get(i * batchSize + j).input, data.get(i * batchSize + j).expect));
						c += this.costEntropy(data.get(i * batchSize + j).input, data.get(i * batchSize + j).expect);
					}
					wb = wb.divide(batchSize);
					this.update(wb, rate);
				}
				
			} else {
				
				for (int i = 0; i < batchNum - 1; ++i) {
					ArrayList<Matrix> w = new ArrayList<Matrix>();
					ArrayList<Matrix> b = new ArrayList<Matrix>();
					
					for (int p = 1; p < layers.length; ++p) {
						w.add(new Matrix(layers[p], layers[p - 1], false));
						b.add(new Matrix(layers[p], 1, false));
					}
					WBPair wb = new WBPair(w, b); 
					for (int j = 0; j < batchSize; ++j) {
						wb = wb.add(this.backPropogateSCE(data.get(i * batchSize + j).input, data.get(i * batchSize + j).expect));
						c += this.costEntropy(data.get(i * batchSize + j).input, data.get(i * batchSize + j).expect);
					}
					wb = wb.divide(batchSize);
					this.update(wb, rate);
				}
				
				for (int p = 0; p < lastBatchSize; ++p) {
					ArrayList<Matrix> w = new ArrayList<Matrix>();
					ArrayList<Matrix> b = new ArrayList<Matrix>();
					
					for (int i = 1; i < layers.length; ++i) {
						w.add(new Matrix(layers[i], layers[i - 1], false));
						b.add(new Matrix(layers[i], 1, false));
					}
					WBPair wb = new WBPair(w, b); 
					
					wb = wb.add(this.backPropogateSCE(data.get((batchNum - 1) * batchSize + p).input, data.get((batchNum - 1) * batchSize + p).expect));
					c += this.costEntropy(data.get((batchNum - 1) * batchSize + p).input, data.get((batchNum - 1) * batchSize + p).expect);
					
					wb = wb.divide(lastBatchSize);
					this.update(wb, rate);
				}
				
			}
			System.out.print("\nEpoch " + z + " completed. The average cost is " + (c / size) + "\n");
			this.test(testData);
		}
		System.out.print("\nTraining completed.\n");
	}
	
	public void trainStochasticSCE(ArrayList<TrainingData> data, int batchSize, double rate, int epoch) {
		for (int z = 0; z < epoch; ++z) {
			Collections.shuffle(data);
			int size = data.size();
			int batchNum = size / batchSize;
			int lastBatchSize = size % batchSize;
			double c = 0;
			if (lastBatchSize == 0) {
				
				for (int i = 0; i < batchNum; ++i) {
					ArrayList<Matrix> w = new ArrayList<Matrix>();
					ArrayList<Matrix> b = new ArrayList<Matrix>();
					
					for (int p = 1; p < layers.length; ++p) {
						w.add(new Matrix(layers[p], layers[p - 1], false));
						b.add(new Matrix(layers[p], 1, false));
					}
					WBPair wb = new WBPair(w, b); 
					for (int j = 0; j < batchSize; ++j) {
						wb = wb.add(this.backPropogateSCE(data.get(i * batchSize + j).input, data.get(i * batchSize + j).expect));
						c += this.costEntropy(data.get(i * batchSize + j).input, data.get(i * batchSize + j).expect);
					}
					wb = wb.divide(batchSize);
					this.update(wb, rate);
				}
				
			} else {
				
				for (int i = 0; i < batchNum - 1; ++i) {
					ArrayList<Matrix> w = new ArrayList<Matrix>();
					ArrayList<Matrix> b = new ArrayList<Matrix>();
					
					for (int p = 1; p < layers.length; ++p) {
						w.add(new Matrix(layers[p], layers[p - 1], false));
						b.add(new Matrix(layers[p], 1, false));
					}
					WBPair wb = new WBPair(w, b); 
					for (int j = 0; j < batchSize; ++j) {
						wb = wb.add(this.backPropogateSCE(data.get(i * batchSize + j).input, data.get(i * batchSize + j).expect));
						c += this.costEntropy(data.get(i * batchSize + j).input, data.get(i * batchSize + j).expect);
					}
					wb = wb.divide(batchSize);
					this.update(wb, rate);
				}
				
				for (int p = 0; p < lastBatchSize; ++p) {
					ArrayList<Matrix> w = new ArrayList<Matrix>();
					ArrayList<Matrix> b = new ArrayList<Matrix>();
					
					for (int i = 1; i < layers.length; ++i) {
						w.add(new Matrix(layers[i], layers[i - 1], false));
						b.add(new Matrix(layers[i], 1, false));
					}
					WBPair wb = new WBPair(w, b); 
					
					wb = wb.add(this.backPropogateSCE(data.get((batchNum - 1) * batchSize + p).input, data.get((batchNum - 1) * batchSize + p).expect));
					c += this.costEntropy(data.get((batchNum - 1) * batchSize + p).input, data.get((batchNum - 1) * batchSize + p).expect);
					
					wb = wb.divide(lastBatchSize);
					this.update(wb, rate);
				}
				
			}
			System.out.print("\nEpoch " + z + " completed. The average cost is " + (c / size) + "\n");
		}
		System.out.print("\nTraining completed.\n");
	}
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	public void test(ArrayList<TrainingData> data) {
		int total = data.size();
		int correct = 0;
		for (int i = 0; i < total; ++i) {
			if (this.feedForward(data.get(i).input).maxIndex(data.get(i).expect)) {
				++correct;
				
				
			} else {
				//System.out.print("\n Case: " + i + " failed.\n");
			}
		}
		System.out.print("\nNetwork testing completed.\n");
		System.out.print("Total validation cases: " + total + "\n");
		System.out.print("Passed validation cases: " + correct + "\n");
		System.out.print("Accuracy: " + (((double)correct/total) * 100) + " %" + "\n");
		
	}
	
	public static class WBPair {
		ArrayList<Matrix> weights;
		ArrayList<Matrix> biases;
		
		public WBPair(ArrayList<Matrix> w, ArrayList<Matrix> b) {
			weights = w;
			biases = b;
		}
		
		public Matrix getWeightsAt(int index) {
			return weights.get(index);
		}
		
		public Matrix getBiasesAt(int index) {
			return biases.get(index);
		}
		
		public WBPair add(WBPair wb) {
			if ((wb.weights.size() != this.weights.size()) || (wb.biases.size() != this.biases.size())) {
				System.out.print("Invalid WBPair input for WBPair addition.\n");
				return null;
			}
			for (int m = 0; m < weights.size(); ++m) {
				if ((wb.weights.get(m).getRowNum() != this.weights.get(m).getRowNum()) || (wb.weights.get(m).getColNum() != this.weights.get(m).getColNum())) {
					System.out.print("Invalid WBPair input for WBPair addition.\n");
					return null;
				}
			}
			for (int n = 0; n < biases.size(); ++n) {
				if ((wb.biases.get(n).getRowNum() != this.biases.get(n).getRowNum()) || (wb.biases.get(n).getRowNum() != this.biases.get(n).getRowNum())) {
					System.out.print("Invalid WBPair input for WBPair addition.\n");
					return null;
				}
			}
			ArrayList<Matrix> resultW = new ArrayList<Matrix>();
			ArrayList<Matrix> resultB = new ArrayList<Matrix>();
		
			for (int i = 0; i < weights.size(); ++i) {
				resultW.add(weights.get(i).add(wb.weights.get(i)));
			}
			for (int j = 0; j < biases.size(); ++j) {
				resultB.add(biases.get(j).add(wb.biases.get(j)));
			}
			
			return new WBPair(resultW, resultB);
			
		}
		
		public WBPair divide(double val) {
			if (val == 0) {
				System.out.print("Invalid value for WBPair division.\n");
				return null;
			}
			WBPair result = this;
			for (int i = 0; i < this.weights.size(); ++i) {
				result.weights.set(i, this.weights.get(i).mult(1 / val));
			}
			for (int j = 0; j < this.biases.size(); ++j) {
				result.biases.set(j, this.biases.get(j).mult(1 / val));
			}
			
			return result;
		}
	}
	
	
	
	
	

}
