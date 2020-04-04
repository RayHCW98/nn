package com.ray.neuralnetwork;

import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.ArrayList;

import mnistReader.MnistMatrix;

public class Matrix implements java.io.Serializable{
	private int rowNum;
	private int colNum;
	private ArrayList<ArrayList<Double>> matrix;
	private int label;
	
	public Matrix(int row, int col, boolean random) {
		rowNum = row;
		colNum = col;
		matrix = new ArrayList<ArrayList<Double>>();
		if (random) {
			for (int i = 0; i < col; ++i) {
				ArrayList<Double> vec = new ArrayList<Double> ();
				for (int j = 0; j < row; ++j) {
					vec.add(Math.random());
				}
				matrix.add(vec);
			}
		} else {
			for (int i = 0; i < col; ++i) {
				ArrayList<Double> vec = new ArrayList<Double> ();
				for (int j = 0; j < row; ++j) {
					vec.add((double) 0);
				}
				matrix.add(vec);
			}
		}
	}
	
	public Matrix(String val) {
		this(val.toCharArray().length, 1, false);
		char[] digits = val.toCharArray();
		for (int i = 0; i < digits.length; ++i) {
			this.setEntry(i, 0, Character.getNumericValue(digits[i]));
		}
		
	}
	
	public Matrix(MnistMatrix mmatrix) {
		this(mmatrix.getNumberOfRows(), mmatrix.getNumberOfColumns(), false);
		for (int i = 0; i < rowNum; ++i) {
			for (int j = 0; j < colNum; ++j) {
				this.setEntry(i, j, mmatrix.getValue(i, j));
			}
		}
	}
	
	public Matrix(int entryNum, int index, double val) {
		this(entryNum, 1, false);
		this.setEntry(index, 0, val);
	}
	
	public Matrix(Matrix mat2) {
		this(mat2.getRowNum(), mat2.getColNum(), false);
	}
	
	public int getRowNum() {
		return rowNum;
	}
	
	public int getColNum() {
		return colNum;
	}
	
	public void setLabel(int val) {
		this.label = val;
	}
	
	public int getLabel() {
		return this.label;
	}
	
	public ArrayList<Double> getCol(int index) {
		return matrix.get(index);
	}
	
	public double getEntry(int rowIndex, int colIndex) {
		return this.getCol(colIndex).get(rowIndex);
	}
	
	public void setEntry(int rowIndex, int colIndex, double val) {
		matrix.get(colIndex).set(rowIndex, val);
	}
	
	public Matrix mult(Matrix mat2) {
		if (colNum != mat2.getRowNum()) {
			System.out.print("Invalid row/column numbers for matrix multiplication!");
			return null;
		}
		Matrix result = new Matrix(rowNum, mat2.getColNum(), false);
		double accu = 0;
		for (int i = 0; i < rowNum; ++i) {
			for (int j = 0; j < mat2.getColNum(); ++j) {
				for (int m = 0; m < colNum; ++m) {
					accu += this.getEntry(i, m) * mat2.getEntry(m, j);
				}
				result.setEntry(i, j, accu);
				accu = 0;
			}
		}
		return result;
	}
	
	public Matrix mult(double val) {
		Matrix result = new Matrix(rowNum, colNum, false);
		for (int i = 0; i < rowNum; ++i) {
			for (int j = 0; j < colNum; ++j) {
				result.setEntry(i, j, val * this.getEntry(i, j));
			}
		}
		return result;
	}
	
	public Matrix add(Matrix mat2) {
		if ((colNum != mat2.getColNum()) || (rowNum != mat2.rowNum)) {
			System.out.print("Invalid row/column numbers for matrix addition!");
			return null;
		}
		Matrix result = new Matrix(rowNum, colNum, false);
		for (int i = 0; i < colNum; ++i) {
			for (int j = 0; j < rowNum; ++j) {
				result.setEntry(j, i, this.getEntry(j, i) + mat2.getEntry(j, i));
			}
		}
		return result;
	}
	
	public Matrix minus(Matrix mat2) {
		if ((colNum != mat2.getColNum()) || (rowNum != mat2.rowNum)) {
			System.out.print("Invalid row/column numbers for matrix substraction!");
			return null;
		}
		Matrix result = new Matrix(rowNum, colNum, false);
		for (int i = 0; i < colNum; ++i) {
			for (int j = 0; j < rowNum; ++j) {
				result.setEntry(j, i, this.getEntry(j, i) - mat2.getEntry(j, i));
			}
		}
		return result;
	}
	
	public Matrix hadamard(Matrix mat2) {
		if ((colNum != mat2.getColNum()) || (rowNum != mat2.rowNum)) {
			System.out.print("Invalid row/column numbers for matrix hadamard!");
			return null;
		}
		Matrix result = new Matrix(rowNum, colNum, false);
		for (int i = 0; i < colNum; ++i) {
			for (int j = 0; j < rowNum; ++j) {
				result.setEntry(j, i, this.getEntry(j, i) * mat2.getEntry(j, i));
			}
		}
		return result;
	}
	
	public Matrix transpose() {
		Matrix transpose = new Matrix(colNum, rowNum, false);
		for (int i = 0; i < rowNum; ++i) {
			for (int j = 0; j < colNum; ++j) {
				transpose.setEntry(j, i, this.getEntry(i, j));
			}
		}
		return transpose;
	}
	
	public void printMatrix() {
		System.out.print("\n\n");
		for (int i = 0; i < rowNum; ++i) {
			System.out.print("|");
			for (int j = 0; j < colNum; ++j) {
				System.out.print(matrix.get(j).get(i) + " ");
			}
			System.out.print("|\n");
		}
		System.out.print("\n\n");
	}
	
	public Matrix sigmoid() {
		Matrix result = new Matrix(this.getRowNum(), this.getColNum(), false);
		for (int i = 0; i < colNum; ++i) {
			for (int j = 0; j < rowNum; ++j) {
				result.setEntry(j, i, sigmoid(this.getEntry(j, i)));
			}
		}
		return result;
	}
	
	public Matrix sigmoidPrime() {
		Matrix result = new Matrix(this.getRowNum(), this.getColNum(), false);
		for (int i = 0; i < colNum; ++i) {
			for (int j = 0; j < rowNum; ++j) {
				result.setEntry(j, i, sigmoidPrime(this.getEntry(j, i)));
			}
		}
		return result;
	}
	
	public boolean equal(Matrix mat2) {
		if ((rowNum != mat2.getRowNum()) || (colNum != mat2.getColNum())) {
			return false;
		}
		for (int i = 0; i < rowNum; ++i) {
			for (int j = 0; j < colNum; ++j) {
				if (this.getEntry(i, j) != mat2.getEntry(i, j)) {
					return false;
				}
			}
		}
		return true;
	}
	
	public boolean closeTo(Matrix mat2) {
		if ((rowNum != mat2.getRowNum()) || (colNum != mat2.getColNum())) {
			return false;
		}
		Matrix diff = this.minus(mat2);
		double cost = 0;
		for (int i = 0; i < rowNum; ++i) {
			for (int j = 0; j < colNum; ++j) {
				cost += Math.pow(diff.getEntry(i, j), 2);
			}
		}
		if (cost < 0.01) {
			return true;
		}
		return false;
	}
	
	public Matrix vectorize() {
		Matrix vec = new Matrix(this.rowNum * this.colNum, 1, false);
		for (int i = 0; i < rowNum; ++i) {
			for (int j = 0; j < colNum; ++j) {
				vec.setEntry(i * colNum + j, 0, this.getEntry(i, j));
			}
		}
		return vec;
	}
	
	public static ArrayList<TrainingData> readData(String dataFilePath, String labelFilePath) throws IOException {

        DataInputStream dataInputStream = new DataInputStream(new BufferedInputStream(new FileInputStream(dataFilePath)));
        int magicNumber = dataInputStream.readInt();
        int numberOfItems = dataInputStream.readInt();
        int nRows = dataInputStream.readInt();
        int nCols = dataInputStream.readInt();

        System.out.println("magic number is " + magicNumber);
        System.out.println("number of items is " + numberOfItems);
        System.out.println("number of rows is: " + nRows);
        System.out.println("number of cols is: " + nCols);

        DataInputStream labelInputStream = new DataInputStream(new BufferedInputStream(new FileInputStream(labelFilePath)));
        int labelMagicNumber = labelInputStream.readInt();
        int numberOfLabels = labelInputStream.readInt();

        System.out.println("labels magic number is: " + labelMagicNumber);
        System.out.println("number of labels is: " + numberOfLabels);

        ArrayList<TrainingData> data = new ArrayList<TrainingData>();

        assert numberOfItems == numberOfLabels;

        for(int i = 0; i < numberOfItems; i++) {
            Matrix matrix = new Matrix(nRows, nCols, false);
            matrix.setLabel(labelInputStream.readUnsignedByte());
            for (int r = 0; r < nRows; r++) {
                for (int c = 0; c < nCols; c++) {
                    matrix.setEntry(r, c, ((double)dataInputStream.readUnsignedByte()) / 255);
                }
            }
            data.add(new TrainingData(matrix.vectorize(), new Matrix(10, matrix.getLabel(), 1)));
        }
        dataInputStream.close();
        labelInputStream.close();
        return data;
    }
	/*
	private static double sigmoid(double val) {
		return Math.max(val, 0);
	}
	private static double sigmoidPrime(double val) {
		if (val < 0) {
			return 0;
		} else {
			return 1;
		}
	}
	*/
	
	private static double sigmoid(double val) {
		return 1 / (1 + Math.exp(-val));
	}
	
	private static double sigmoidPrime(double val) {
		return sigmoid(val) * (1 - sigmoid(val));
	}
	
}