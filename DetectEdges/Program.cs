using System;
using System.Runtime.InteropServices;
using System.Drawing;
using Accord.DataSets;
using Accord.Imaging.Filters;
using Accord.Controls;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;


namespace DetectEdges
{
    class Program
    {
        // Import the libSystem shared library and define the method
        // corresponding to the native function.
        [DllImport("/Users/leo/Projects/DetectEdges/DetectEdges/libfind_edges.dylib")]
        private static extern int round_int(double num);

        static void Main(string[] args)
        {
            ////invoke function in the dll file
            //// Invoke the function
            //double num1 = 3.4;
            //double num2 = 3.6;
            //int result1 = round_int(num1);
            //int result2 = round_int(num2);
            //Console.WriteLine(result1);
            //Console.WriteLine(result2);


            ////get image
            //string filename = "/Users/leo/Projects/DetectEdges/DetectEdges/lena5.jpg";
            //Bitmap image1 = Accord.Imaging.Image.FromFile(filename);
            //image1.Save("/Users/leo/Downloads/abc1.png");


            ////Gaussian Filter
            //GaussianBlur gaussianBlur = new GaussianBlur();
            //Bitmap gBimage1 = gaussianBlur.Apply(image1);
            //gBimage1.Save("/Users/leo/Downloads/abc2.png");


            ////toMatrix
            //Double[,] matrix1;
            //Accord.Imaging.Converters.ImageToMatrix imageToMatrix = new Accord.Imaging.Converters.ImageToMatrix();
            //imageToMatrix.Convert(image1, out matrix1);
            //printMatrix(matrix1, 0, 20, 0, 20);


            ////toGray
            //Grayscale grayscale = new Grayscale(0.2989, 0.5870, 0.114);
            //Bitmap grayImage1 = grayscale.Apply(image1);
            //grayImage1.Save("/Users/leo/Downloads/abc3.png");


            ////gradient
            //Double[,] matrix2 = Program.gradient(matrix1);
            //Console.WriteLine("------------------------------------------------------");
            //printMatrix(matrix2, 0, 20, 0, 20);


            ////matrixToImage
            //Accord.Imaging.Converters.MatrixToImage matrixToImage = new Accord.Imaging.Converters.MatrixToImage();
            //Bitmap image2;
            //matrixToImage.Convert(matrix2, out image2);
            //image2.Save("/Users/leo/Downloads/abc4.png");

            ////pad a matrix
            //Matrix<Double> matrix_test = Matrix.Build.Random(3, 4);
            //printMatrix(matrix_test, 0, 4, 0, 3);
            //Matrix<Double> matrix_result = pad_matrix(matrix_test, new double[] { 3, 4 });
            //printMatrix(matrix_result, 0, matrix_result.ColumnCount, 0, matrix_result.RowCount);

            ////trim a matrix
            //Matrix<Double> matrix_test2 = Matrix.Build.Random(3, 4);
            //printMatrix(matrix_test2, 0, 4, 0, 3);
            //Matrix<Double> matrix_result2 = trim_matrix(matrix_test2, new double[] { 1, 3 });
            //printMatrix(matrix_result2, 0, matrix_result2.ColumnCount, 0, matrix_result2.RowCount);

            ////conv2
            //double[,] x = { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } };
            //Matrix<Double> matrix_input = Matrix.Build.DenseOfArray(x);
            //double[,] y = { { -1, -2, -1 }, { 0, 0, 0 }, { 1, 2, 1 } };
            //Matrix<Double> matrix_filter = Matrix.Build.DenseOfArray(y);
            //Matrix<Double> result3 = conv2(matrix_input, matrix_filter, "same");
            //printMatrix(result3, 0, result3.ColumnCount, 0, result3.RowCount);

            ////convolve_2
            ////double[,] x = { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } };
            //Matrix<Double> matrix_input1 = Matrix.Build.DenseOfArray(x);
            ////double[,] y = { { -1, -2, -1 }, { 0, 0, 0 }, { 1, 2, 1 } };
            //Matrix<Double> matrix_filter1 = Matrix.Build.DenseOfArray(y);
            //Matrix<Double> result4 = convolve_2(matrix_input1, matrix_filter1, 1);
            //printMatrix(result4, 0, result4.RowCount, 0, result4.ColumnCount);

            ////gauss
            //double[,] matrix = { { 1, 2,3}, { 4, 5, 6}, { 7, 8, 9} };
            //Matrix<Double> x = Matrix.Build.DenseOfArray(matrix);
            //int std = 2;
            //Console.WriteLine(gauss(x, std));

            ////d2gauss
            //Console.WriteLine(d2gauss(3,2,3,2,0.5,5.5));
            //Console.WriteLine(d2gauss(17, 1.7321, 1, 1, 0, 0.2303));

            ////setvalues
            //Console.WriteLine(setvalues(3));

            ////rotateMatrix90
            //double[,] matrix = { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } };
            //Matrix<Double> x = Matrix.Build.DenseOfArray(matrix);
            //Console.WriteLine(RotateMatrix90MultiTimes(x, 4));
            //Console.WriteLine(RotateMatrix90MultiTimes(x, 3));
            //Console.WriteLine(RotateMatrix90MultiTimes(x, 2));
            //Console.WriteLine(RotateMatrix90MultiTimes(x, 1));

            ////scalespace
            //double[,] matrix = { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } };
            //double[][,] result;
            //result = scalespace(matrix, 3, 0);
            //for (int i = 0; i <= 3; i++)
            //{
            //    Matrix<Double> m = Matrix<Double>.Build.DenseOfArray(result[i]);
            //    Console.WriteLine("{0} :  {1}", i, m);
            //}

            ////g1steer
            //double[,] g1x = { { 1, 2, 3 }, { 4, 0, 6 }, { 0, 8, 0 } };
            //double[,] g1y = { { 9, 0, 7 }, { 6, 5, 0 }, { 3, 0, 1 } };
            //double[][,] result = g1steer(g1x, g1y);
            //double[,] g1dir = result[0];
            //double[,] g1mag = result[1];
            //printMatrix(g1dir, 0, g1dir.GetLength(0), 0, g1dir.GetLength(1));
            //printMatrix(g1mag, 0, g1mag.GetLength(0), 0, g1mag.GetLength(1));

            //g1scale
            //double[,] g1mag2 = {
            //    { 9, 0, 7.6, 9, 0, 7.6, 9, 0, 7.6, 9, 0, 7.6, 9, 0, 7.6, 9, 0, 7.6, 9, 0, 7.6, 9, 0, 7.6, 9, 0, 7.6 },
            //    { 7.2, 0, 0, 7.2, 0, 0, 7.2, 0, 0, 7.2, 0, 0, 7.2, 0, 0, 7.2, 0, 0, 7.2, 0, 0, 7.2, 0, 0, 7.2, 0, 0 },
            //    { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
            //    { 9, 0, 7.6, 9, 0, 7.6, 9, 0, 7.6, 9, 0, 7.6, 9, 0, 7.6, 9, 0, 7.6, 9, 0, 7.6, 9, 0, 7.6, 9, 0, 7.6 },
            //    { 7.2, 0, 0, 7.2, 0, 0, 7.2, 0, 0, 7.2, 0, 0, 7.2, 0, 0, 7.2, 0, 0, 7.2, 0, 0, 7.2, 0, 0, 7.2, 0, 0 },
            //    { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
            //    { 9, 0, 7.6, 9, 0, 7.6, 9, 0, 7.6, 9, 0, 7.6, 9, 0, 7.6, 9, 0, 7.6, 9, 0, 7.6, 9, 0, 7.6, 9, 0, 7.6 },
            //    { 7.2, 0, 0, 7.2, 0, 0, 7.2, 0, 0, 7.2, 0, 0, 7.2, 0, 0, 7.2, 0, 0, 7.2, 0, 0, 7.2, 0, 0, 7.2, 0, 0 },
            //    { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
            //    { 9, 0, 7.6, 9, 0, 7.6, 9, 0, 7.6, 9, 0, 7.6, 9, 0, 7.6, 9, 0, 7.6, 9, 0, 7.6, 9, 0, 7.6, 9, 0, 7.6 },
            //    { 7.2, 0, 0, 7.2, 0, 0, 7.2, 0, 0, 7.2, 0, 0, 7.2, 0, 0, 7.2, 0, 0, 7.2, 0, 0, 7.2, 0, 0, 7.2, 0, 0 },
            //    { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
            //    { 9, 0, 7.6, 9, 0, 7.6, 9, 0, 7.6, 9, 0, 7.6, 9, 0, 7.6, 9, 0, 7.6, 9, 0, 7.6, 9, 0, 7.6, 9, 0, 7.6 },
            //    { 7.2, 0, 0, 7.2, 0, 0, 7.2, 0, 0, 7.2, 0, 0, 7.2, 0, 0, 7.2, 0, 0, 7.2, 0, 0, 7.2, 0, 0, 7.2, 0, 0 },
            //    { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
            //    { 9, 0, 7.6, 9, 0, 7.6, 9, 0, 7.6, 9, 0, 7.6, 9, 0, 7.6, 9, 0, 7.6, 9, 0, 7.6, 9, 0, 7.6, 9, 0, 7.6 },
            //    { 7.2, 0, 0, 7.2, 0, 0, 7.2, 0, 0, 7.2, 0, 0, 7.2, 0, 0, 7.2, 0, 0, 7.2, 0, 0, 7.2, 0, 0, 7.2, 0, 0 },
            //    { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }};
            //double[,] g1dir2 = {
            //    { 9, 0, 7.6, 9, 0, 7.6, 9, 0, 7.6, 9, 0, 7.6, 9, 0, 7.6, 9, 0, 7.6, 9, 0, 7.6, 9, 0, 7.6, 9, 0, 7.6 },
            //    { 7.2, 0, 0, 7.2, 0, 0, 7.2, 0, 0, 7.2, 0, 0, 7.2, 0, 0, 7.2, 0, 0, 7.2, 0, 0, 7.2, 0, 0, 7.2, 0, 0 },
            //    { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
            //    { 9, 0, 7.6, 9, 0, 7.6, 9, 0, 7.6, 9, 0, 7.6, 9, 0, 7.6, 9, 0, 7.6, 9, 0, 7.6, 9, 0, 7.6, 9, 0, 7.6 },
            //    { 7.2, 0, 0, 7.2, 0, 0, 7.2, 0, 0, 7.2, 0, 0, 7.2, 0, 0, 7.2, 0, 0, 7.2, 0, 0, 7.2, 0, 0, 7.2, 0, 0 },
            //    { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
            //    { 9, 0, 7.6, 9, 0, 7.6, 9, 0, 7.6, 9, 0, 7.6, 9, 0, 7.6, 9, 0, 7.6, 9, 0, 7.6, 9, 0, 7.6, 9, 0, 7.6 },
            //    { 7.2, 0, 0, 7.2, 0, 0, 7.2, 0, 0, 7.2, 0, 0, 7.2, 0, 0, 7.2, 0, 0, 7.2, 0, 0, 7.2, 0, 0, 7.2, 0, 0 },
            //    { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
            //    { 9, 0, 7.6, 9, 0, 7.6, 9, 0, 7.6, 9, 0, 7.6, 9, 0, 7.6, 9, 0, 7.6, 9, 0, 7.6, 9, 0, 7.6, 9, 0, 7.6 },
            //    { 7.2, 0, 0, 7.2, 0, 0, 7.2, 0, 0, 7.2, 0, 0, 7.2, 0, 0, 7.2, 0, 0, 7.2, 0, 0, 7.2, 0, 0, 7.2, 0, 0 },
            //    { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
            //    { 9, 0, 7.6, 9, 0, 7.6, 9, 0, 7.6, 9, 0, 7.6, 9, 0, 7.6, 9, 0, 7.6, 9, 0, 7.6, 9, 0, 7.6, 9, 0, 7.6 },
            //    { 7.2, 0, 0, 7.2, 0, 0, 7.2, 0, 0, 7.2, 0, 0, 7.2, 0, 0, 7.2, 0, 0, 7.2, 0, 0, 7.2, 0, 0, 7.2, 0, 0 },
            //    { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
            //    { 9, 0, 7.6, 9, 0, 7.6, 9, 0, 7.6, 9, 0, 7.6, 9, 0, 7.6, 9, 0, 7.6, 9, 0, 7.6, 9, 0, 7.6, 9, 0, 7.6 },
            //    { 7.2, 0, 0, 7.2, 0, 0, 7.2, 0, 0, 7.2, 0, 0, 7.2, 0, 0, 7.2, 0, 0, 7.2, 0, 0, 7.2, 0, 0, 7.2, 0, 0 },
            //    { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }};
            //double[][,] result = g1scale(g1mag2, g1dir2, 3, 0.1, 0);
            //double[,] g1mag1 = result[0];
            //double[,] g1dir1 = result[1];
            //double[,] g1sc1 = result[2];
            //Console.WriteLine("g1mag1:");
            //printMatrix(g1mag1, 0, g1mag1.GetLength(0), 0, g1mag1.GetLength(1));
            //Console.WriteLine("g1dir1:");
            //printMatrix(g1dir1, 0, g1dir1.GetLength(0), 0, g1dir1.GetLength(1));
            //Console.WriteLine("g1sc1:");
            //printMatrix(g1sc1, 0, g1sc1.GetLength(0), 0, g1sc1.GetLength(1));

            ////read_gy
            string g1scaleval = "1";
            string fm2 = ".ascii";
            //string[] kern1_str = System.IO.File.ReadAllLines("/Users/leo/Projects/DetectEdges/DetectEdges/filters/g1y" + g1scaleval + fm2);
            //double[,] kern1 = read_gy(kern1_str);
            //printMatrix(kern1, 0, kern1.GetLength(0), 0, kern1.GetLength(1));

            // gradient
            String filename = "/Users/leo/Projects/DetectEdges/DetectEdges/lena5.jpg";
            Bitmap im = Accord.Imaging.Image.FromFile(filename);
            double[,] matrix_im;
            Accord.Imaging.Converters.ImageToMatrix imageToMatrix = new Accord.Imaging.Converters.ImageToMatrix();
            imageToMatrix.Convert(im, out matrix_im);
            //printMatrix(matrix_im, 0, matrix_im.GetLength(0), 0, matrix_im.GetLength(1));
            Grayscale grayscale = new Grayscale(0.2989, 0.5870, 0.114);
            Bitmap gray_im = grayscale.Apply(im);
            gray_im.Save("/Users/leo/Downloads/gray.png");
            double[,] matrix_grayim;
            
            imageToMatrix.Convert(gray_im, out matrix_grayim);
            for (int i = 0; i < matrix_grayim.GetLength(0); i++)
            {
                for (int j = 0; j < matrix_grayim.GetLength(1); j++)
                {
                    matrix_grayim[i, j] = matrix_grayim[i, j] * 255;
                }
            }
            printMatrix(matrix_grayim, 0, 20, 0, 20);

           // double[][,] gauss_a = scalespace(matrix_grayim, 5, 1);
           // //printMatrix(gauss_a[5], 299, 310, 299, 310);
           //// double[][,] gr = gradient(5, 1, gauss_a, 0);

           // //kern1
           // string[] kern1_str = System.IO.File.ReadAllLines("/Users/leo/Projects/DetectEdges/DetectEdges/filters/gy" + g1scaleval + fm2);
           // Matrix<Double> kern1 = Matrix.Build.DenseOfArray(read_gy(kern1_str));
           // Matrix<Double> rc1 = convolve_2(Matrix.Build.DenseOfArray(gauss_a[5]), kern1, 0);
            
           // //kern2
           // string[] kern2_str = System.IO.File.ReadAllLines("/Users/leo/Projects/DetectEdges/DetectEdges/filters/g1x" + g1scaleval + fm2);
           // Matrix<Double> kern2 = Matrix.Build.DenseOfArray(read_gx(kern2_str));
           // Matrix<Double> rc2 = convolve_2(rc1, kern2, 0);
            
           // //kern3
           // string[] kern3_str = System.IO.File.ReadAllLines("/Users/leo/Projects/DetectEdges/DetectEdges/filters/gx" + g1scaleval + fm2);
           // Matrix<Double> kern3 = Matrix.Build.DenseOfArray(read_gx(kern3_str));
           // //Console.WriteLine("------kern3-----");
           // Matrix<Double> rc3 = convolve_2(Matrix.Build.DenseOfArray(gauss_a[5]), kern3, 0);
           
           // //kern4
           // string[] kern4_str = System.IO.File.ReadAllLines("/Users/leo/Projects/DetectEdges/DetectEdges/filters/g1y" + g1scaleval + fm2);
           // Matrix<Double> kern4 = Matrix.Build.DenseOfArray(read_gy(kern4_str));
           // //Console.WriteLine("------kern4-----");
           // //printMatrix(kern4, 0, kern4.RowCount, 0, kern4.ColumnCount);
           // Matrix<Double> rc4 = convolve_2(rc3, kern4, 0);
           // Console.WriteLine("------rc4--------");      
           // double[][,] gsteer = g1steer(rc2.ToArray(), rc4.ToArray());
           // Console.WriteLine("m2 row:{0}, m2 col:{1}", gsteer[0].GetLength(0), gsteer[1].GetLength(1));
           // printMatrix(gsteer[0], 299, 310, 299, 310);
           // Console.WriteLine("d2 row:{0}, d2 col:{1}", gsteer[1].GetLength(0), gsteer[1].GetLength(1));
           // printMatrix(gsteer[1], 299, 310, 299, 310);









            //printMatrix(kern1, 0, kern1.RowCount, 0, kern1.ColumnCount);
            //Matrix<double> result = convolve_2(Matrix.Build.DenseOfArray(matrix_grayim), kern1, 0);
            //printMatrix(result, 299, 310, 299, 310);
            //Console.WriteLine("row:{0}, col:{1}", matrix_grayim.GetLength(0), matrix_grayim.GetLength(1));
            //Accord.Imaging.Converters.MatrixToImage matrixToImage = new Accord.Imaging.Converters.MatrixToImage();
            //matrixToImage.Convert(matrix_grayim, out gray_im);
            //gray_im.Save("/Users/leo/Downloads/gray2.png");
            ////Console.WriteLine("length0:{0}, length1:{1}", matrix_grayim.GetLength(0), matrix_grayim.GetLength(1));
            //int scale = 5;
            //double[][,] gauss_imgs = scalespace(matrix_grayim, scale, 1);
            //double[][,] gradient_result = gradient(5, 1, gauss_imgs, 0);
            //Bitmap image_g1mag;
            //double[,] g1dir = gradient_result[1];
            //for (int i = 0; i < matrix_grayim.GetLength(0); i++)
            //{
            //    for (int j = 0; j < matrix_grayim.GetLength(1); j++)
            //    {
            //        g1dir[i, j] = g1dir[i, j] / 255;
            //    }
            //}
            //matrixToImage.Convert(g1dir, out image_g1mag);
            //image_g1mag.Save("/Users/leo/Downloads/g1dir.png");
            //printMatrix(gradient_result[0], 0, gradient_result[0].GetLength(0), 0, gradient_result[1].GetLength(1));



            ////printMatrix(rc4, 0, rc4.RowCount, 0, rc4.ColumnCount);
            //double[][,] g1steer_result = g1steer(rc2.ToArray(), rc4.ToArray());
            //double[,] m2 = g1steer_result[0];
            //double[,] d2 = g1steer_result[1];
            //printMatrix(m2, 0, m2.GetLength(0), 0, m2.GetLength(1));
            //Bitmap image_m2;
            //matrixToImage.Convert(m2, out image_m2);
            //image_m2.Save("/Users/leo/Downloads/m2.png");


        }


        //##############################################################################
        //#              Gradient Function of Matlab-Don't use!!!                      #
        //#                                                                            #
        //#  gradient calculates the central difference for interior data points.      #
        //#  The interior gradient values, G(:, j), are                                #
        //#  G(:, j) = 0.5*(A(:,j+1) - A(:, j-1));                                     #
        //#                                                                            #
        //##############################################################################
        static Double[,] gradient(Double[,] two_d_array)
        {
            Matrix<Double> matrix = Matrix<Double>.Build.DenseOfArray(two_d_array);
            Matrix<Double> result = Matrix<Double>.Build.Dense(matrix.RowCount,matrix.ColumnCount);
            int columnNum = matrix.ColumnCount;
            int rowNum = matrix.RowCount;
           
            for (int i = 0; i < columnNum; i++)
            {
                for (int j = 0; j < rowNum; j++)
                {
                    //gradient calculates values along the edges of the matrix with single - sided differences:
                    if (i == 0)
                    {
                        //Console.Write("matrix[j,0]: "+matrix[j, 0] + "---");
                        //Console.WriteLine("matrix[j,1]: "+matrix[j, 1]);
                        result[j, 0] = matrix[j, 1] - matrix[j, 0];

                    }
                    //gradient calculates values along the edges of the matrix with single - sided differences:
                    else if (i == columnNum - 1)
                    {
                        result[j, columnNum - 1] = matrix[j, columnNum - 1] - matrix[j, columnNum - 2];
                    }
                    //not along the edges
                    else
                    {
                        result[j, i] = 0.5 * (matrix[j, i + 1] - matrix[j, i - 1]);
                    }
                }
            }
            // convert matrix to 2d array
            Double[,] resultArr = result.ToArray();
            
            return resultArr;
        }

        //##############################################################################
        //#                               print matrix                                 #
        //#     colend = columnCount                                                   #
        //#     rowend = rowCount                                                      #       
        //##############################################################################

        static void printMatrix(Matrix<Double> matrix, int rowstart, int rowend, int colstart, int colend)
        {
            int rowLength = matrix.RowCount;
            int colLength = matrix.ColumnCount;
            for (int i = rowstart; i < rowend; i++)
            {
                for (int j = colstart; j < colend; j++)
                {
                    Console.Write(string.Format("{0:0.000} ", matrix[i, j]));
                }
                Console.Write(Environment.NewLine + Environment.NewLine);
            }
        }

        static void printMatrix(Double[,] matrix, int rowstart, int rowend, int colstart, int colend)
        {
            
            for (int i = rowstart; i < rowend; i++)
            {
                for (int j = colstart; j < colend; j++)
                {
                    if (matrix == null)
                    {
                        Console.WriteLine("matrix to print cannot be null");
                    } else
                    {
                        Console.Write(string.Format("{0:0.000} ", matrix[i, j]));

                    }
                }
                Console.Write(Environment.NewLine + Environment.NewLine);
            }
        }

        static void printMatrix(int[,] matrix, int rowstart, int rowend, int colstart, int colend)
        {

            for (int i = rowstart; i < rowend; i++)
            {
                for (int j = colstart; j < colend; j++)
                {
                    if (matrix == null)
                    {
                        Console.WriteLine("matrix to print cannot be null");
                    }
                    else
                    {
                        Console.Write(string.Format("{0} ", matrix[i, j]));

                    }
                }
                Console.Write(Environment.NewLine + Environment.NewLine);
            }
        }

        //##############################################################################
        //#                               convolve_2                                   #
        //##############################################################################
        static Matrix<Double> convolve_2(Matrix<Double> mimg, Matrix<Double> filter, int bc)
        {
            Console.WriteLine("enter convolve_2");

            Matrix<Double> pad_img, cimg;
            int k;
            double[] dimention = { filter.RowCount, filter.ColumnCount };

            if (bc == 0)
            {
                pad_img = pad_matrix(mimg, dimention);
                cimg = conv2(pad_img, filter,"same");
                cimg = trim_matrix(cimg, dimention);
        
            }
            else
            {
                cimg = conv2(mimg, filter, "same");

                if (dimention[0] < dimention[1])
                {
                    k = (int)Math.Floor(0.5 * filter.ColumnCount);
                    for (int i = 0; i < cimg.RowCount; i++)
                    {
                        for (int j = 0; j < k; j++)
                        {
                            cimg[i, j] = 0;
                        }

                        for (int j = cimg.ColumnCount - k; j < cimg.ColumnCount; j++)
                        {
                            cimg[i, j] = 0;
                        }
                    }
                }
                else
                {
                    k = (int)Math.Floor(0.5 * filter.RowCount);
                    for (int j = 0; j < cimg.ColumnCount; j++)
                    {
                        for (int i = 0; i < k; i++)
                        {
                            cimg[i, j] = 0;
                        }

                        for (int i = cimg.RowCount - k; i < cimg.RowCount; i++)
                        {
                            cimg[i, j] = 0;
                        }
                    }

                }
            }
            return cimg;
        }
        //##############################################################################
        //#                               function to pad matrix                       #
        //##############################################################################
        static Matrix<Double> pad_matrix(Matrix<Double> matrix, double[] dimention)
        {
            int k;
            Matrix<Double> ad1_init, ad2_init, ad1, ad2, result;
            // num of row < num of column
            if (dimention[0] < dimention[1])
            {
                k = (int)Math.Floor(0.5 * dimention[1]);
                ad1_init = matrix.Column(0).ToColumnMatrix();//generate column vector at column index 0, toColumnMatrix: convert vector to matrix
                ad1 = ad1_init;
                for (int i = 0; i < k - 1; i++)
                {
                    ad1 = ad1.Append(ad1_init);//concatenate the column
                }
                ad2_init = matrix.Column(matrix.ColumnCount - 1).ToColumnMatrix();
                ad2 = ad2_init;
                for (int i = 0; i < k - 1; i++)
                {
                    ad2 = ad2.Append(ad2_init);//concatenate the column
                }
                result = ad1.Append(matrix).Append(ad2);
            }
            else
            {
                k = (int)Math.Floor(0.5 * dimention[0]);
                ad1_init = matrix.Row(0).ToRowMatrix();
                ad1 = ad1_init;
                for (int i = 0; i < k - 1; i++)
                {
                    ad1 = ad1.Stack(ad1_init);//concatenate the row
                }
                ad2_init = matrix.Row(matrix.RowCount - 1).ToRowMatrix();
                ad2 = ad2_init;
                for (int i = 0; i < k - 1; i++)
                {
                    ad2 = ad2.Stack(ad2_init);//concatenate the row
                }
                result = ad1.Stack(matrix).Stack(ad2);
            }
            return result;
        }

        //##############################################################################
        //#                               function to trim matrix                      #
        //##############################################################################
        static Matrix<Double> trim_matrix(Matrix<Double> matrix, double[] dimention)
        {
            int k;
            Matrix<Double> result;
            if (dimention[0] < dimention[1])
            {
                k = (int)Math.Floor(0.5 * dimention[1]);
                result = Matrix<Double>.Build.Dense(matrix.RowCount, matrix.ColumnCount - 2 * k);
                for (int i = k; i < matrix.ColumnCount-k; i++)
                {    
                    result.SetColumn(i-k, matrix.Column(i));
                }
            }
            else
            {
                k = (int)Math.Floor(0.5 * dimention[0]);
                result = Matrix<Double>.Build.Dense(matrix.RowCount - 2 * k, matrix.ColumnCount);
                for (int i = k; i < matrix.RowCount - k; i++)
                { 
                    result.SetRow(i-k, matrix.Row(i));
                }
            }
            return result;
        }

        //##############################################################################
        //#                               conv2                                        #
        //##############################################################################
        static Matrix<Double> conv2(Matrix<Double> A, Matrix<Double> B, string s)
        {
            
            int rowA = A.RowCount;
            int colA = A.ColumnCount;
            int rowB = B.RowCount;
            int colB = B.ColumnCount;
            int rowResult = Math.Max(rowA + rowB - 1, Math.Max(rowA, rowB));
            int colResult = Math.Max(colA + colB - 1, Math.Max(colA, colB));
            Matrix<Double> result = Matrix.Build.Dense(rowResult, colResult);
            for (int r1 = 0; r1 < rowResult; r1++)
            {
                for (int r2 = 0; r2 < colResult; r2++)
                {
                    for (int a1 = 0; a1 < rowA; a1++)
                    {
                        int b1 = r1 - a1;
                        if(b1 < 0 || b1 >= rowB)
                        {
                            continue;
                        }

                        for (int a2 = 0; a2 < colA; a2++)
                        {
                            int b2 = r2 - a2;
                            if (b2 < 0 || b2 >= colB) {
                                continue;
                            }

                            result[r1, r2] = result[r1, r2] + A[a1, a2] * B[b1, b2];
                        }
                    }
                }
            }
            if (!s.Equals("same")) 
                return result;
            else
            {
                int rowStartIndex = (int) Math.Floor((rowResult - rowA) / 2.0);
                int colStartIndex = (int) Math.Floor((colResult - colA) / 2.0);
                Matrix<Double> result2 = Matrix.Build.Dense(rowA, colA);
                for (int i = rowStartIndex; i < rowStartIndex + rowA; i++)
                {
                    for (int j = colStartIndex; j < colStartIndex + colA; j++)
                    {
                        result2[i - rowStartIndex, j - colStartIndex] = result[i, j];
                    }
                }
                return result2;
            }
        }

        //##############################################################################
        //#                               d2gauss                                      #
        //#         returns a 2-d Gaussian filter with kernal attributes:              #
        //#           size:       n1* n2                                               #
        //#           theta:      CCW-angle tkernat filter rotated                     #
        //#           sigma1:     standard deviation of 1st gaussian                   #
        //#           sigma2:     standard deviation of 2nd gaussian                   #
        //##############################################################################
        static Matrix<Double> d2gauss(int n1, double std1, int n2, double std2, double theta, double max1)
        {
            Console.WriteLine("enter d2gauss");

            double[] n1_vector = new double[n1];
            double[] n2_vector = new double[n2];
            //result of meshgrid
            double[,] I, J;
            
            //initialize n1_vector
            for (int i = 0; i < n1; i++)
            {
                n1_vector[i] = i + 1;
            }
            //initialize n2_vector
            for (int i = 0; i < n2; i++)
            {
                n2_vector[i] = i + 1;
            }
            Tuple<double[,], double[,]> tuple = Accord.Math.Matrix.MeshGrid<Double>(n2_vector, n1_vector);
            I = tuple.Item1;
            J = tuple.Item2;
            double[,] It = new double[I.GetLength(0), I.GetLength(1)];
            double[,] Jt = new double[J.GetLength(0), J.GetLength(1)];
            for (int i = 0; i < I.GetLength(0); i++)
            {
                for (int j = 0; j < I.GetLength(1); j++)
                {
                    It[i, j] = I[i, j] - (n2 + 1) / 2;
                    Jt[i, j] = J[i, j] - (n1 + 1) / 2;
                }
            }
            //transpose It, Jt
            //double[,] It_transpose = Accord.Math.Matrix.Transpose<double>(It);
            //double[,] Jt_transpose = Accord.Math.Matrix.Transpose<double>(Jt);

            double[,] u1 = new double[It.GetLength(0),It.GetLength(1)], u2 = new double[It.GetLength(0), It.GetLength(1)];
            for (int i = 0; i < It.GetLength(0); i++)
            {
                for (int j = 0; j < It.GetLength(1); j++)
                {
                    u1[i, j] = Math.Cos(theta) * Jt[i, j] - Math.Sin(theta) * It[i, j];
                    u2[i, j] = Math.Sin(theta) * Jt[i, j] + Math.Cos(theta) * It[i, j];
                }
            }
            Matrix<Double> u1_matrix = Matrix<Double>.Build.DenseOfArray(u1);
            Matrix<Double> u2_matrix = Matrix<Double>.Build.DenseOfArray(u2);
            Matrix<Double> g1 = gauss(u1_matrix, std1);
            Matrix<Double> g2 = gauss(u2_matrix, std2);
            Matrix<Double> kern = g1.PointwiseMultiply(g2);

            //Normalise the kernal and confine to limits:
            double max2 = Double.MinValue;
            double sum = 0;
            //get sum
            for (int i = 0; i < kern.RowCount; i++)
            {
                for (int j = 0; j < kern.ColumnCount; j++)
                {
                    sum += kern[i, j] * kern[i, j];
                }
            }
            
            for (int i = 0; i < kern.RowCount; i++)
            {
                for (int j = 0; j < kern.ColumnCount; j++)
                {
                    kern[i, j] = (kern[i, j] / Math.Sqrt(sum)); 
                }
            }
            //get max2
            for (int i = 0; i < kern.RowCount; i++)
            {
                for (int j = 0; j < kern.ColumnCount; j++)
                {
                    if (max2 < kern[i, j])
                    {
                        max2 = kern[i, j];
                    }
                }
            }

            for (int i = 0; i < kern.RowCount; i++)
            {
                for (int j = 0; j < kern.ColumnCount; j++)
                {
                    kern[i, j] = (kern[i, j] / (max2/max1));
                }
            }

            return kern;
        }

        //##############################################################################
        //#                               gauss                                        #
        //##############################################################################
        static Matrix<Double> gauss(Matrix<Double> x, double std)
        {
            Console.WriteLine("enter gauss");

            Matrix<Double> result = Matrix<Double>.Build.Dense(x.RowCount, x.ColumnCount);
            for (int i = 0; i < x.RowCount; i++)
            {
                for (int j = 0; j < x.ColumnCount; j++)
                {
                    result[i, j] = Math.Exp(-(x[i, j] * x[i, j]) / (2 * std * std)) / (std * Math.Sqrt(2 * Math.PI));
                }
            }
            return result;
        }

        //########################################################################################
        //#                               setvalues
        //#         Set values for generating 2d Gaussian filter according to 
        //#         input scale.
        //#
        //#          Input:    scale
        //# 
        //#         Output:   stdd:       std dev'n along width (along height assumed to be 1)
        //#                   size:       width of output = #columns (height assumed to be 1)
        //########################################################################################
        static Tuple<double, int> setvalues(double scale)
        {
            Console.WriteLine("enter setvalue");

            double stdd;
            int sizz;
            if (scale < 3)
            {
                stdd = 1;
            }
            else
            {
                stdd = Math.Sqrt(Math.Pow(Math.Pow(2, scale - 2), 2) - 1);
            }
            sizz  = 2 * (int)Math.Ceiling(4.6 * stdd) + 1;
            Tuple<double, int> result = new Tuple<double, int>(stdd, sizz);
            return result;
        }

        //#######################################################################################
        //#
        //#                          scalespace  
        //#
        //#     Create a series of Gaussian blurred images according to a 
        //#     given maximum scale.
        //#
        //#     Input:    mimg:         image for convolution w/ Gaussian filter
        //#           maxscale:         maximum scale value
        //#          conv_type:         convolution type flag
        //#
        //#     Output:   blurred_imgs:   maxscale blurred images
        //#
        //#######################################################################################
        static double[][,] scalespace(double[,] mimg, int maxscale, int conv_type)
        {
            Console.WriteLine("enter scalespace");

            double[][,] blurred_imgs = new double[maxscale+1][,];
            double stdd;
            int sizz;

            //tuple to hold stdd, sizz
            Tuple<double, int> values_for_gauss;
            Matrix<Double> kern, c1mimg, cres;
            blurred_imgs[0] = new double[mimg.GetLength(0),mimg.GetLength(1)]; 

            for (int scale = 1; scale <= maxscale; scale++)
            {
                if (scale < 3)
                {
                    // Image is unblurred: 
                    blurred_imgs[scale] = mimg;
                }
                else
                {
                    // Set values for generating Gaussian filter at given scale:
                    values_for_gauss = setvalues(scale);
                    stdd = values_for_gauss.Item1;
                    sizz = values_for_gauss.Item2;
                    kern = d2gauss(sizz, stdd, 1, 1, 0, 1/(stdd * Math.Sqrt(2*Math.PI)));
                    Matrix<Double> mimg_matrix = Matrix<Double>.Build.DenseOfArray(mimg);
                    c1mimg = convolve_2(mimg_matrix, kern, conv_type);
                    cres = convolve_2(c1mimg, RotateMatrix90MultiTimes(kern, 3), conv_type);

                    blurred_imgs[scale] = cres.ToArray();
                }
            }
            return blurred_imgs;
        }


        //##############################################################################
        //#                               Rotate Matrix 90 one time                    #
        //##############################################################################
        static Matrix<Double> RotateMatrix90(Matrix<Double> oldMatrix)
        {
            Matrix<Double> newMatrix = Matrix<Double>.Build.Dense(oldMatrix.ColumnCount, oldMatrix.RowCount);
            int newColumn, newRow = 0;
            for (int oldColumn = oldMatrix.ColumnCount - 1; oldColumn >= 0; oldColumn--)
            {
                newColumn = 0;
                for (int oldRow = 0; oldRow < oldMatrix.RowCount; oldRow++)
                {
                    newMatrix[newRow, newColumn] = oldMatrix[oldRow, oldColumn];
                    newColumn++;
                }
                newRow++;
            }
            return newMatrix;
        }

        //##############################################################################
        //#                               Rotate Matrix 90 multi times                 #
        //##############################################################################
        static Matrix<Double> RotateMatrix90MultiTimes(Matrix<Double> matrix, int times)
        {
            for (int i = 0; i < times; i++)
            {
                Matrix<Double> matrix2 = RotateMatrix90(matrix);
                matrix = matrix2;
            }
            return matrix;
        }

        //##############################################################################
        //#                               g1steer
        //#     g1steer - Computes magnitude and direction of the gradient  of the 
        //#     luminance function based on x and y basis functions for 1st Gaussian 
        //#     derivative.
        //# 
        //#       
        //#        Input:  g1x - X basis for Gaussian gradient
        //#                g1y - Y basis for Gaussian gradient
        //#        Output: g1mag - Gradient Magnitude Estimate
        //#                g1dir - Gradient Direction Estimate
        //##############################################################################
        static double[][,] g1steer(double[,] g1x, double[,] g1y)
        {
            Console.WriteLine("enter g1steer");
            double[][,] result = new double[2][,];
            //initial g1mag with zeros
            Matrix<Double> g1mag = Matrix.Build.Dense(g1x.GetLength(0), g1x.GetLength(1));
            //initial g1dir with 4
            Matrix<Double> g1dir = Matrix.Build.Dense(g1x.GetLength(0), g1x.GetLength(1), 4.0);
            for (int i = 0; i < g1x.GetLength(0); i++)
            {
                for (int j = 0; j < g1x.GetLength(1); j++)
                {
                    if ((! double.Equals(g1x[i, j], 0.0)) && (! double.Equals(g1y[i, j], 0.0)))
                    {
                        g1dir[i, j] = Math.Atan2(-g1y[i,j], g1x[i,j]);
                        g1mag[i, j] = Math.Sqrt(g1x[i, j] * g1x[i, j] + g1y[i, j] * g1y[i, j]);
                    }
                }
            }

            result[0] = g1mag.ToArray();
            result[1] = g1dir.ToArray();
            return result;
        }

        //##############################################################################
        //#
        //# g1scale(g1mag1, g1dir1, g1mag2, g1dir2, g1scale1, scale, noise, b_est)
        //#
        //# Purpose: Augments multi-scale Gaussian Gradient maps with significant 
        //#		estimates at a new scale.Pixels for which the magnitude 
        //#		of the Gaussian gradient is under threshold in the multi-scale 
        //#		map(gradient magnitude input 1) but over threshold 
        //# 		at the new scale(gradient magnitude input 2) are updated 
        //#		with the gradient magnitude, direction and scale value of 
        //#		the new scale.
        //#
        //# Input:  g1mag1    - Multi-scale Gaussian gradient magnitude image
        //#         g1dir1    - Multi-scale Gaussian gradient direction image
        //#         g1mag2    - Gaussian gradient magnitude image at new scale
        //#         g1dir2    - Gaussian gradient direction image at new scale
        //#         g1scale1  - Multi-scale scale map
        //#         g1scale2  - Scale of new gradient estimates
        //#         noise     - Estimated sensor noise
        //#         b_est     - Derivatives near boundary estimated by reflecting
        //#                           intensity function.
        //# Output: g1mag1    - Integrated multi-scale gradient magnitude map
        //#         g1dir1    - Integrated multi-scale gradient direction map
        //#         g1scale1  - Integrated multi-scale scale map
        //#
        //##############################################################################
        static double[][,] g1scale(double[,] g1mag2, double[,] g1dir2, int scale, double noise, int b_est)
        {
            Console.WriteLine("enter g1scale");
            double[][,] result = new double[3][,];
            int krad;
            double[,] g1mag1, g1dir1, g1sc1;
            //initialize g1mag1 g1dir1 g1sc1
            g1mag1 = Matrix.Build.Dense(g1mag2.GetLength(0), g1mag2.GetLength(1)).ToArray();
            g1dir1 = Matrix.Build.Dense(g1mag2.GetLength(0), g1mag2.GetLength(1)).ToArray();
            g1sc1 = Matrix.Build.Dense(g1mag2.GetLength(0), g1mag2.GetLength(1)).ToArray();
            double[] norms12 = { 0.765, 0.199, 0.0499, 0.0125, 0.00312, 0.00078 };
            double thresh = 5.6 * noise * norms12[scale];
            if (scale < 3 | b_est == 1)
            {
                krad = 1;
            }
            else
            {
                krad = (int) Math.Ceiling(4.6 * Math.Sqrt(Math.Pow(2, 2*(scale - 2)) - 1));
            }
            
            if (scale == 1)
            {
                g1mag1 = Matrix.Build.Dense(g1mag2.GetLength(0), g1mag2.GetLength(1)).ToArray();
                g1dir1 = Matrix.Build.Dense(g1mag2.GetLength(0), g1mag2.GetLength(1), 4.0).ToArray();
                g1sc1 = Matrix.Build.Dense(g1mag2.GetLength(0), g1mag2.GetLength(1)).ToArray();
                for (int i = 0; i < g1mag2.GetLength(0); i++)
                {
                    for (int j = 0; j < g1mag2.GetLength(1); j++)
                    {
                        if (g1mag2[i, j] >= thresh)
                        {
                            g1mag1[i, j] = g1mag2[i, j];
                            g1dir1[i, j] = g1dir2[i, j];
                            g1sc1[i, j] = scale;
                        }
                    }
                }
                result[0] = g1mag1;
                result[1] = g1dir1;
                result[2] = g1sc1;
                return result;
            }
            else
            {
                int[] sz = new int[2];
                sz[0] = g1mag2.GetLength(0);
                sz[1] = g1mag2.GetLength(1);
                int[,] matrix_i, matrix_j;
                int v1_length = sz[0] - 2 * krad;                
                int v2_length = sz[1] - 2 * krad;
                //matrix_i or matrix_j are not empty
                if (v1_length > 1 && v2_length > 1)
                {
                    int[] v1 = new int[v1_length];
                    int[] v2 = new int[v2_length];
                    for (int i = 0; i < v1_length; i++)
                    {
                        v1[i] = krad + i;
                    }
                    for (int i = 0; i < v2_length; i++)
                    {
                        v2[i] = krad + i;
                    }

                    Tuple<int[,], int[,]> tuple = Accord.Math.Matrix.MeshGrid<int>(v1, v2);
                    matrix_i = tuple.Item1;
                    matrix_j = tuple.Item2;
                    //int[,] matrix_i_transpose = Accord.Math.Matrix.Transpose<int>(matrix_i);
                    //int[,] matrix_j_transpose = Accord.Math.Matrix.Transpose<int>(matrix_j);

                    //scaleMatrix : smat
                    double[,] scaleMatrix = new double[g1sc1.GetLength(0) - 2*krad, g1sc1.GetLength(1) - 2*krad];
                    for (int i = 0; i < g1sc1.GetLength(0) - 2*krad; i++)
                    {
                        for (int j = 0; j < g1sc1.GetLength(1) -2*krad; j++)
                        {
                            scaleMatrix[i, j] = g1sc1[krad + i, krad + j];
                        }
                    }
                    //magMatrix : mmat
                    double[,] magMatrix = new double[g1mag2.GetLength(0)-2*krad, g1mag2.GetLength(1)-2*krad];
                    for (int i = 0; i < g1mag2.GetLength(0) - 2*krad; i++)
                    {
                        for (int j = 0; j < g1mag2.GetLength(1) - 2*krad; j++)
                        {
                            magMatrix[i, j] = g1mag2[krad + i, krad + j];
                        }
                    }

                    for (int i = 0; i < scaleMatrix.GetLength(0); i++)
                    {
                        for (int j = 0; j < scaleMatrix.GetLength(1); j++)
                        {
                            //f = find((smat==0) & (mmat>=thresh));
                            if ((double.Equals(scaleMatrix[i, j], 0.0)) && (magMatrix[i, j] >= thresh))
                            {
                                //g1mag1(K(f)) = g1mag2(K(f));
                                //g1dir1(K(f)) = g1dir2(K(f));
                                //g1sc1(K(f)) = scale;
                                g1mag1[matrix_i[i, j], matrix_j[i, j]] = g1mag2[matrix_i[i, j], matrix_j[i, j]];
                                g1dir1[matrix_i[i, j], matrix_j[i, j]] = g1dir2[matrix_i[i, j], matrix_j[i, j]];
                                g1sc1[matrix_i[i, j], matrix_j[i, j]] = scale;
                            }
                        }
                    }
                }
                else
                {
                    Console.WriteLine("matrix_i or matrix_j is empty");
                }
                result[0] = g1mag1;
                result[1] = g1dir1;
                result[2] = g1sc1;
                return result;
            }
        }
        //##############################################################################
        //#                               read gx
        //#
        //#         input string[] - read from file
        //#         output double[,] 
        //##############################################################################
        static double[,] read_gx(string[] kern_str)
        {
            string[] kern_split = kern_str[0].Split(' ');
            int kern_col = kern_split.GetLength(0) - 1;
            double[,] kern = new double[1, kern_col];
            for (int i = 0; i < kern_col; i++)
            {
                String s = kern_split[i].Trim();
                kern[0, i] = double.Parse(s);
            }
            return kern;
        }

        //##############################################################################
        //#                               read gy
        //#
        //#         input string[] - read from file
        //#         output double[,] 
        //##############################################################################
        static double[,] read_gy(string[] kern_str)
        {
            int kern_row = kern_str.GetLength(0);
            double[,] kern = new double[kern_row, 1];
            for (int i = 0; i < kern_row; i++)
            {
                kern[i, 0] = Convert.ToDouble(kern_str[i]);
            }
            return kern;
        }

        //#####################################################################################
        //#
        //#         gradient(maxscale, noise, gauss_a, conv_type, filtpath)
        //#         Computes non-zero gradient in the luminance function by 
        //# 
        //#####################################################################################
        static double[][,] gradient(int maxscale, double noise, double[][,] gauss_a, int conv_type)
        {
            Console.WriteLine("enter gradient");

            double[][,] g1scale_result = new double[3][,];
            String g1scaleval, fm2 = ".ascii";
            for (int scale = 1; scale <= maxscale; scale++)
            {
                if (scale == 1)
                {
                    g1scaleval = "05";
                }
                else
                {
                    g1scaleval = "1";
                }

                Matrix<Double> mimg = Matrix.Build.DenseOfArray(gauss_a[scale]);

                //^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                //^     Compute response of first derivative Gaussian filter
                //^     to the blurred image in an arbitrary direction:
                //^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

                //kern1
                string[] kern1_str = System.IO.File.ReadAllLines("/Users/leo/Projects/DetectEdges/DetectEdges/filters/gy" + g1scaleval + fm2);
                Matrix<Double> kern1 = Matrix.Build.DenseOfArray(read_gy(kern1_str));
                Matrix<Double> rc1 = convolve_2(mimg, kern1, conv_type);

                //kern2
                string[] kern2_str = System.IO.File.ReadAllLines("/Users/leo/Projects/DetectEdges/DetectEdges/filters/g1x" + g1scaleval + fm2);
                Matrix<Double> kern2 = Matrix.Build.DenseOfArray(read_gx(kern2_str));
                Matrix<Double> rc2 = convolve_2(rc1, kern2, conv_type);

                //kern3
                string[] kern3_str = System.IO.File.ReadAllLines("/Users/leo/Projects/DetectEdges/DetectEdges/filters/gx" + g1scaleval + fm2);
                Matrix<Double> kern3 = Matrix.Build.DenseOfArray(read_gx(kern3_str));
                Matrix<Double> rc3 = convolve_2(mimg, kern3, conv_type);

                //kern4
                string[] kern4_str = System.IO.File.ReadAllLines("/Users/leo/Projects/DetectEdges/DetectEdges/filters/g1y" + g1scaleval + fm2);
                Matrix<Double> kern4 = Matrix.Build.DenseOfArray(read_gy(kern4_str));
                Matrix<Double> rc4 = convolve_2(rc3, kern4, conv_type);

                //^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                //^     Calculate magnitude and direction of the gradient:
                //^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                double[][,] g1steer_result = g1steer(rc2.ToArray(), rc4.ToArray());
                double[,] m2 = g1steer_result[0];

                double[,] d2 = g1steer_result[1];

                //^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                //^     Augment multi-scale Gaussian Gradient maps:
                //^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                g1scale_result = g1scale(m2, d2, scale, noise, 0);

            }
            return g1scale_result;
        }




    }
}
