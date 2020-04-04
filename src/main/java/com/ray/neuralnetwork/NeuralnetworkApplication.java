package com.ray.neuralnetwork;

import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.util.ArrayList;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.CommandLineRunner;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.core.env.Environment;
import org.springframework.stereotype.Component;


@SpringBootApplication
@Component
public class NeuralnetworkApplication implements CommandLineRunner{
	@Autowired
	private Environment env;
	public static void main(String[] args) {
		SpringApplication.run(NeuralnetworkApplication.class, args);
	}
	
	@Override
	public void run(String... args) throws Exception {
		/*
		int[] layers = new int[] {1, 1};
		NeuralNetwork net = new NeuralNetwork(layers);
		Matrix input = new Matrix(1, 1, false);
		Matrix expect = new Matrix(1, 1, false);
		input.setEntry(0, 0, 1);
		expect.setEntry(0, 0, 0);
		TrainingData datum = new TrainingData(input, expect);
		ArrayList<TrainingData> data = new ArrayList<TrainingData>();
		data.add(datum);
		net.printNetworkWeights();
		net.printNetworkBiases();
		net.trainStochastic(data, 1, 3, 100);
		net.printNetworkWeights();
		net.printNetworkBiases();
		*/
		
		
		ArrayList<TrainingData> trainData = Matrix.readData(env.getProperty("training.data"), env.getProperty("training.label"));
		ArrayList<TrainingData> testData = Matrix.readData(env.getProperty("testing.data"), env.getProperty("testing.label"));

		
		
		int[] layers = new int[]{784, 30, 10};
		NeuralNetwork net = new NeuralNetwork(layers);
		
		//net.printNetworkBiases();

		net.trainStochastic(trainData, 1000, 300000000, 30);
		net.test(testData);
		
		
		try{
			 FileOutputStream fileOut =
			 new FileOutputStream("/Users/wanghaochen/Desktop/NN/Neural_Network_Warehouse/nn.ser");
			 ObjectOutputStream out = new ObjectOutputStream(fileOut);
			 out.writeObject(net);
			 out.close();
			 fileOut.close();
			 System.out.printf("/Users/wanghaochen/Desktop/NN/Neural_Network_Warehouse/nn.ser");
	    }
		catch(IOException i)
	    {
	          i.printStackTrace();
	    }
	    
		/*int[] layers = new int[] {16, 4};
		NeuralNetwork net = new NeuralNetwork(layers);
		ArrayList<TrainingData> data = TrainingData.testTraining;

		net.train(data, 113, 10000);
		*/

	}
	
	

}
