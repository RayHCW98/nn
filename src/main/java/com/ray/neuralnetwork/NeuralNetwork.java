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
	
	public WBPair backPropogate(Matrix input, Matrix expect) {
		ArrayList<Matrix> errors = new ArrayList<Matrix>();
		ArrayList<Matrix> weights_delta = new ArrayList<Matrix>();
		ArrayList<Matrix> biases_delta = new ArrayList<Matrix>();
		
		int len = layers.length;
		
		Matrix output = feedForward(input);
		Matrix gradientC = output.minus(expect);
		Matrix errorL = gradientC.hadamard(this.feedForwardRawAt(input, len - 1).sigmoidPrime());
		errors.add(errorL);
		for (int j = len - 2; j > 0; --j) {
			errors.add(weights.get(j).transpose().mult(errors.get(len - 2 - j)).hadamard(feedForwardRawAt(input, j).sigmoidPrime()));
		}
		Collections.reverse(errors);
		
		biases_delta = errors;
		
		for (int m = 0; m < len - 1; ++m) {
			weights_delta.add(errors.get(m).mult(this.feedForwardOutputAt(input, m).transpose()));
		}
		
//		for (int n = 0; n < len - 1; ++n) {
//			weights.set(n, weights.get(n).minus(weights_delta.get(n).mult(rate)));
//			biases.set(n, biases.get(n).minus(biases_delta.get(n).mult(rate)));
//		}
		
		return new WBPair(weights_delta, biases_delta);
	}
	
	public double cost(Matrix input, Matrix expect) {
		Matrix output = feedForward(input);
		Matrix diff = expect.minus(output);
		double result = 0;
		for (int i = 0; i < diff.getRowNum(); ++i) {
			result += Math.pow(diff.getEntry(i, 0), 2);
		}
		return result;
	}
	
	public void update(WBPair wb, double rate) {
		int len = layers.length;
		for (int n = 0; n < len - 1; ++n) {
			weights.set(n, weights.get(n).minus(wb.getWeightsAt(n).mult(rate)));
			biases.set(n, biases.get(n).minus(wb.getBiasesAt(n).mult(rate)));
		}
	}
	
	public void train(ArrayList<TrainingData> data, double rate, int epoch) {
		int size = data.size();
		for (int i = 0; i < epoch; ++i) {
			double c = 0;
			for (int j = 0; j < data.size(); ++j) {
				WBPair wb = backPropogate(data.get(j).input, data.get(j).expect);
				c += this.cost(data.get(j).input, data.get(j).expect);
				update(wb, rate);
			}
			//System.out.print("\nEpoch " + i + " completed. The average cost is " + (c / data.size()) + "\n");
			System.out.print("\nEpoch " + i + " completed. The average cost is " + (c / size) + "\n");
		}
		System.out.print("\nTraining completed.\n");
	}
	
	public void trainStochastic(ArrayList<TrainingData> data, int batchSize, double rate, int epoch) {
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
						wb = wb.add(this.backPropogate(data.get(i * batchSize + j).input, data.get(i * batchSize + j).expect));
						c += this.cost(data.get(i * batchSize + j).input, data.get(i * batchSize + j).expect);
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
						wb = wb.add(this.backPropogate(data.get(i * batchSize + j).input, data.get(i * batchSize + j).expect));
						c += this.cost(data.get(i * batchSize + j).input, data.get(i * batchSize + j).expect);
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
					
					wb = wb.add(this.backPropogate(data.get((batchNum - 1) * batchSize + p).input, data.get((batchNum - 1) * batchSize + p).expect));
					c += this.cost(data.get((batchNum - 1) * batchSize + p).input, data.get((batchNum - 1) * batchSize + p).expect);
					
					wb = wb.divide(lastBatchSize);
					this.update(wb, rate);
				}
				
			}
			System.out.print("\nEpoch " + z + " completed. The average cost is " + (c / size) + "\n");
		}
		System.out.print("\nTraining (Stochastic Gradient Descent) completed.\n");
	}
	
	public void test(ArrayList<TrainingData> data) {
		int total = data.size();
		int correct = 0;
		for (int i = 0; i < total; ++i) {
			if (this.feedForward(data.get(i).input).closeTo(data.get(i).expect)) {
				++correct;
				System.out.print("\nTest " + i + " passed!\n");
				System.out.print("Expect: ");
				data.get(i).expect.printMatrix();
				System.out.print("Output: ");
				this.feedForward(data.get(i).input).printMatrix();
				
			} else {
				System.out.print("\nTest " + i + " failed!\n");
				System.out.print("Expect: ");
				data.get(i).expect.printMatrix();
				System.out.print("Output: ");
				this.feedForward(data.get(i).input).printMatrix();
			}
		}
		System.out.print("\nNetwork testing completed.\n");
		System.out.print("Total validation cases: " + total + "\n");
		System.out.print("Passed validation cases: " + correct + "\n");
		System.out.print("Accuracy: " + ((double)correct/total) + " %" + "\n");
		
	}
	
	private class WBPair {
		private ArrayList<Matrix> weights;
		private ArrayList<Matrix> biases;
		
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
