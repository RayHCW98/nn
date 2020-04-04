package com.ray.neuralnetwork;

import java.util.ArrayList;

public class TrainingData {
	Matrix input;
	Matrix expect;
	/*static ArrayList<TrainingData> testTraining =  new ArrayList<TrainingData> ();
	static {
		Matrix zero = new Matrix("1000000000000000");
		
		Matrix one = new Matrix("0100000000000000");
		
		Matrix two = new Matrix("0010000000000000");

		Matrix three = new Matrix("0001000000000000");
		
		Matrix four = new Matrix("0000100000000000");
		
		Matrix five = new Matrix("0000010000000000");
		
		Matrix six = new Matrix("0000001000000000");
		
		Matrix seven = new Matrix("0000000100000000");
		
		Matrix eight = new Matrix("0000000010000000");
		
		Matrix nine = new Matrix("0000000001000000");
		
		Matrix ten = new Matrix("0000000000100000");
		
		Matrix ele = new Matrix("0000000000010000");
		
		Matrix twelve = new Matrix("0000000000001000");
		
		Matrix thir = new Matrix("0000000000000100");
		
		Matrix fourt = new Matrix("0000000000000010");
		
		Matrix fif = new Matrix("0000000000000001");
		
		// ********************
		
		Matrix ezero = new Matrix("0000");
		
		Matrix eone = new Matrix("0001");
		
		Matrix etwo = new Matrix("0010");

		Matrix ethree = new Matrix("0011");
		
		Matrix efour = new Matrix("0100");
		
		Matrix efive = new Matrix("0101");
		
		Matrix esix = new Matrix("0110");
		
		Matrix eseven = new Matrix("0111");
		
		Matrix eeight = new Matrix("1000");
		
		Matrix enine = new Matrix("1001");
		
		Matrix eten = new Matrix("1010");
		
		Matrix eele = new Matrix("1011");
		
		Matrix etwelve = new Matrix("1100");
		
		Matrix ethir = new Matrix("1101");
		
		Matrix efourt = new Matrix("1110");
		
		Matrix efif = new Matrix("1111");
		
		testTraining.add(new TrainingData(zero, ezero));
		testTraining.add(new TrainingData(one, eone));
		testTraining.add(new TrainingData(two, etwo));
		testTraining.add(new TrainingData(three, ethree));
		testTraining.add(new TrainingData(four, efour));
		testTraining.add(new TrainingData(five, efive));
		testTraining.add(new TrainingData(six, esix));
		testTraining.add(new TrainingData(seven, eseven));
		testTraining.add(new TrainingData(eight, eeight));
		testTraining.add(new TrainingData(nine, enine));
		testTraining.add(new TrainingData(ten, eten));
		testTraining.add(new TrainingData(ele, eele));
		testTraining.add(new TrainingData(twelve, etwelve));
		testTraining.add(new TrainingData(thir, ethir));
		testTraining.add(new TrainingData(fourt, efourt));
		testTraining.add(new TrainingData(fif, efif));

	} */
	
	public TrainingData(Matrix in, Matrix ex) {
		input = in;
		expect = ex;
	}
}
