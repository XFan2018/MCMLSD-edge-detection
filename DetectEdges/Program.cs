﻿using System;
using System.Runtime.InteropServices;
using System.Drawing;
using Accord.DataSets;
using Accord.Imaging.Filters;
using Accord.Controls;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using MathNet.Numerics.Data.Matlab;
using csmatio.types;
using csmatio.io;
using System.Collections.Generic;
using Accord.Math;
using Matrix = MathNet.Numerics.LinearAlgebra.Double.Matrix;
using MatrixLA = MathNet.Numerics.LinearAlgebra.Matrix<double>;
using Emgu.CV;
using Emgu.CV.Structure;
using Emgu.CV.Util;


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

            ////test derivative2nd

            ////step 1: input image
            //string filename = "/Users/leo/Projects/DetectEdges/DetectEdges/img.jpg";
            //Bitmap im = Accord.Imaging.Image.FromFile(filename);

            ////step2: convert to gray image
            //double[,] matrix_im;
            //Accord.Imaging.Converters.ImageToMatrix imageToMatrix = new Accord.Imaging.Converters.ImageToMatrix();
            //imageToMatrix.Convert(im, out matrix_im);
            //Grayscale grayscale = new Grayscale(0.2989, 0.5870, 0.1140);
            //Bitmap gray_im = grayscale.Apply(im);
            //gray_im.Save("/Users/leo/Downloads/gray.png");
            //double[,] matrix_grayim;

            //imageToMatrix.Convert(gray_im, out matrix_grayim);
            //for (int i = 0; i < matrix_grayim.GetLength(0); i++)
            //{
            //    for (int j = 0; j < matrix_grayim.GetLength(1); j++)
            //    {
            //        matrix_grayim[i, j] = matrix_grayim[i, j] * 255;
            //    }
            //}

            ////step3: compose blurred image
            //double[][,] gauss_a = scalespace(matrix_grayim, 5, 1);
            //double[,] g1mag1 = Matrix.Build.Dense(gauss_a[0].GetLength(0), gauss_a[0].GetLength(1)).ToArray();
            //double[,] g1dir1 = Matrix.Build.Dense(gauss_a[0].GetLength(0), gauss_a[0].GetLength(1)).ToArray();
            //double[,] g1sc1 = Matrix.Build.Dense(gauss_a[0].GetLength(0), gauss_a[0].GetLength(1)).ToArray();

            ////step4: calculate gradient map
            //double[][,] gr = gradient(5, 1, gauss_a, 0);

            //g1mag1 = gr[0];
            //g1dir1 = gr[1];
            //g1sc1 = gr[2];

            ////for (int i = 0; i < gr[0].GetLength(0); i++)
            ////{
            ////    for (int j = 0; j < gr[0].GetLength(1); j++)
            ////    {
            ////        g1mag1[i, j] = g1mag1[i, j] / 255;
            ////        g1dir1[i, j] = g1dir1[i, j] / 255;
            ////        g1sc1[i, j] = g1sc1[i, j] / 255;
            ////    }
            ////}



            ////for (int i = 0; i < gauss_a[0].GetLength(0); i++)
            ////{
            ////    for (int j = 0; j < gauss_a[0].GetLength(1); j++)
            ////    {
            ////        matrix_grayim[i, j] = matrix_grayim[i, j] / 255;
            ////        gauss_a[1][i, j] = gauss_a[1][i, j] / 255;
            ////        gauss_a[2][i, j] = gauss_a[2][i, j] / 255;
            ////        gauss_a[3][i, j] = gauss_a[3][i, j] / 255;
            ////        gauss_a[4][i, j] = gauss_a[4][i, j] / 255;
            ////        gauss_a[5][i, j] = gauss_a[5][i, j] / 255;


            ////    }
            ////}

            ////Bitmap g1mag, g1dir, g1sc, gauss_a1, gauss_a2, gauss_a3, gauss_a4, gauss_a5;
            //Accord.Imaging.Converters.MatrixToImage matrixToImage = new Accord.Imaging.Converters.MatrixToImage();

            ////matrixToImage.Convert(g1mag1, out g1mag);
            ////matrixToImage.Convert(g1dir1, out g1dir);
            ////matrixToImage.Convert(g1sc1, out g1sc);
            ////matrixToImage.Convert(matrix_grayim, out gray_im);
            ////matrixToImage.Convert(gauss_a[1], out gauss_a1);
            ////matrixToImage.Convert(gauss_a[2], out gauss_a2);
            ////matrixToImage.Convert(gauss_a[3], out gauss_a3);
            ////matrixToImage.Convert(gauss_a[4], out gauss_a4);
            ////matrixToImage.Convert(gauss_a[5], out gauss_a5);

            ////g1mag.Save("/Users/leo/Downloads/g1mag1.jpg");
            ////g1dir.Save("/Users/leo/Downloads/g1dir1.jpg");
            ////g1sc.Save("/Users/leo/Downloads/g1sc1.jpg");
            ////gray_im.Save("/Users/leo/Downloads/gray_im.jpg");
            ////gauss_a1.Save("/Users/leo/Downloads/gauss_a1.jpg");
            ////gauss_a2.Save("/Users/leo/Downloads/gauss_a2.jpg");
            ////gauss_a3.Save("/Users/leo/Downloads/gauss_a3.jpg");
            ////gauss_a4.Save("/Users/leo/Downloads/gauss_a4.jpg");
            ////gauss_a5.Save("/Users/leo/Downloads/gauss_a5.jpg");

            //// calculate 2nd derivative map
            //double[][,] d2 = derivative2nd(gr[1], 5, 1, gauss_a, 0);

            //for (int i = 0; i < d2[0].GetLength(0); i++)
            //{
            //    for (int j = 0; j < d2[0].GetLength(1); j++)
            //    {
            //        d2[0][i, j] = d2[0][i, j] / 255;
            //        d2[1][i, j] = d2[1][i, j] / 255;
            //    }
            //}

            //for (int i = 0; i < d2[2].GetLength(0); i++)
            //{
            //    for (int j = 0; j < d2[2].GetLength(1); j++)
            //    {
            //        d2[2][i, j] = d2[2][i, j] / 255;

            //    }
            //}

            //Bitmap g2mag, g2sc, g2all;
            //matrixToImage.Convert(d2[0], out g2mag);
            //matrixToImage.Convert(d2[1], out g2sc);
            //matrixToImage.Convert(d2[2], out g2all);
            //g2mag.Save("/Users/leo/Downloads/g2mag.jpg");
            //g2sc.Save("/Users/leo/Downloads/g2sc.jpg");
            //g2all.Save("/Users/leo/Downloads/g2all.jpg");
            //im.Save("/Users/leo/Downloads/im.jpg");

            ////export c# array to mat file
            //double[][] d2_g2mag = new double[d2[0].GetLength(0)][];
            //for (int i = 0; i < d2[0].GetLength(0); i++)
            //{
            //    d2_g2mag[i] = new double[d2[1].GetLength(1)];
            //    for (int j = 0; j < d2[1].GetLength(1); j++)
            //    {

            //        d2_g2mag[i][j] = d2[0][i, j];
            //    }
            //}
            //MLDouble mlDoubleArray = new MLDouble("Matrix_3_by_3", d2_g2mag);
            //List<MLArray> mlList = new List<MLArray>();
            //mlList.Add(mlDoubleArray);
            //MatFileWriter mfw = new MatFileWriter("/Users/leo/Projects/DetectEdges/DetectEdges/data.mat", mlList, false);

            double[,] arr = { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } };
            double[,] outarr = new double[3, 3];
            GCHandle handle = GCHandle.Alloc(arr, GCHandleType.Pinned);
            IntPtr kernel = handle.AddrOfPinnedObject();
            Mat m = new Mat(3, 3, Emgu.CV.CvEnum.DepthType.Cv64F, 1, kernel, 16);
            CvInvoke.Flip(m, m, Emgu.CV.CvEnum.FlipType.Horizontal);







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
            MatrixLA matrix = MatrixLA.Build.DenseOfArray(two_d_array);
            MatrixLA result = MatrixLA.Build.Dense(matrix.RowCount, matrix.ColumnCount);
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

        static void printMatrix(MatrixLA matrix, int rowstart, int rowend, int colstart, int colend)
        {
            int rowLength = matrix.RowCount;
            int colLength = matrix.ColumnCount;
            for (int i = rowstart; i < rowend; i++)
            {
                for (int j = colstart; j < colend; j++)
                {
                    Console.Write(string.Format("{0:0.0000} ", matrix[i, j]));
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
                    }
                    else
                    {
                        Console.Write(string.Format("{0:0.0000} ", matrix[i, j]));

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
        static MatrixLA convolve_2(MatrixLA mimg, MatrixLA filter, int bc)
        {
            Console.WriteLine("enter convolve_2");

            MatrixLA pad_img, cimg;
            int k;
            double[] dimention = { filter.RowCount, filter.ColumnCount };

            if (bc == 0)
            {
                pad_img = pad_matrix(mimg, dimention);
                cimg = conv2(pad_img, filter, "same");
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
        static MatrixLA pad_matrix(MatrixLA matrix, double[] dimention)
        {
            int k;
            MatrixLA ad1_init, ad2_init, ad1, ad2, result;
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
        static MatrixLA trim_matrix(MatrixLA matrix, double[] dimention)
        {
            int k;
            MatrixLA result;
            if (dimention[0] < dimention[1])
            {
                k = (int)Math.Floor(0.5 * dimention[1]);
                result = MatrixLA.Build.Dense(matrix.RowCount, matrix.ColumnCount - 2 * k);
                for (int i = k; i < matrix.ColumnCount - k; i++)
                {
                    result.SetColumn(i - k, matrix.Column(i));
                }
            }
            else
            {
                k = (int)Math.Floor(0.5 * dimention[0]);
                result = MatrixLA.Build.Dense(matrix.RowCount - 2 * k, matrix.ColumnCount);
                for (int i = k; i < matrix.RowCount - k; i++)
                {
                    result.SetRow(i - k, matrix.Row(i));
                }
            }
            return result;
        }

        //##############################################################################
        //#                               conv2                                        #
        //##############################################################################
        static MatrixLA conv2(MatrixLA A, MatrixLA B, string s)
        {
            Console.Write("enter conv2");
            int rowA = A.RowCount;
            int colA = A.ColumnCount;
            int rowB = B.RowCount;
            int colB = B.ColumnCount;
            int rowResult = Math.Max(rowA + rowB - 1, Math.Max(rowA, rowB));
            int colResult = Math.Max(colA + colB - 1, Math.Max(colA, colB));
            MatrixLA result = Matrix.Build.Dense(rowResult, colResult);
            for (int r1 = 0; r1 < rowResult; r1++)
            {
                for (int r2 = 0; r2 < colResult; r2++)
                {
                    for (int a1 = 0; a1 < rowA; a1++)
                    {
                        int b1 = r1 - a1;
                        if (b1 < 0 || b1 >= rowB)
                        {
                            continue;
                        }

                        for (int a2 = 0; a2 < colA; a2++)
                        {
                            int b2 = r2 - a2;
                            if (b2 < 0 || b2 >= colB)
                            {
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
                int rowStartIndex = (int)Math.Floor((rowResult - rowA) / 2.0);
                int colStartIndex = (int)Math.Floor((colResult - colA) / 2.0);
                MatrixLA result2 = Matrix.Build.Dense(rowA, colA);
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
        //#                               conv2                                        #
        //##############################################################################
        //static MatrixLA conv2emgu(MatrixLA A, MatrixLA B, string s)
        //{
        //    //int rowA = A.RowCount;
        //    //int colA = A.ColumnCount;
        //    //int rowB = B.RowCount;
        //    //int colB = B.ColumnCount;
        //    //int rowResult = Math.Max(rowA + rowB - 1, Math.Max(rowA, rowB));
        //    //int colResult = Math.Max(colA + colB - 1, Math.Max(colA, colB));
        //    //MatrixLA result = Matrix.Build.Dense(rowResult, colResult);
        //    //Point anchor = new Point(colB - colB/2 - 1, rowB - rowB/2 - 1);
        //    //GCHandle handleB = GCHandle.Alloc(B, GCHandleType.Pinned);
        //    //GCHandle handleA = GCHandle.Alloc(A, GCHandleType.Pinned);
        //    //GCHandle handleR = GCHandle.Alloc(result, GCHandleType.Pinned);
        //    //IntPtr kernel = handleB.AddrOfPinnedObject();
        //    //IntPtr src = handleA.AddrOfPinnedObject();
        //    //IntPtr dst = handleR.AddrOfPinnedObject();
        //    //// flip kernel: flip_mode < 0 (e.g. -1) means flipping around both axises
        //    //CvInvoke.Flip(kernel, kernel, -1);
        //    //CvInvoke.Filter2D();
        //    //for (int i = 0; i < rowA; i++)
        //    //{
        //    //    double[] temp = new double[colA];
        //    //    Marshal.Copy(dst, temp, i, colA);
        //    //    for (int j = 0; j < colA; j++)
        //    //    {   
        //    //        result[i, j] = temp[j];
        //    //    }                
        //    //}
        //    //return result;
        //}

        //##############################################################################
        //#                               d2gauss                                      #
        //#         returns a 2-d Gaussian filter with kernal attributes:              #
        //#           size:       n1* n2                                               #
        //#           theta:      CCW-angle tkernat filter rotated                     #
        //#           sigma1:     standard deviation of 1st gaussian                   #
        //#           sigma2:     standard deviation of 2nd gaussian                   #
        //##############################################################################
        static MatrixLA d2gauss(int n1, double std1, int n2, double std2, double theta, double max1)
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

            double[,] u1 = new double[It.GetLength(0), It.GetLength(1)], u2 = new double[It.GetLength(0), It.GetLength(1)];
            for (int i = 0; i < It.GetLength(0); i++)
            {
                for (int j = 0; j < It.GetLength(1); j++)
                {
                    u1[i, j] = Math.Cos(theta) * Jt[i, j] - Math.Sin(theta) * It[i, j];
                    u2[i, j] = Math.Sin(theta) * Jt[i, j] + Math.Cos(theta) * It[i, j];
                }
            }
            MatrixLA u1_matrix = MatrixLA.Build.DenseOfArray(u1);
            MatrixLA u2_matrix = MatrixLA.Build.DenseOfArray(u2);
            MatrixLA g1 = gauss(u1_matrix, std1);
            MatrixLA g2 = gauss(u2_matrix, std2);
            MatrixLA kern = g1.PointwiseMultiply(g2);

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
                    kern[i, j] = (kern[i, j] / (max2 / max1));
                }
            }

            return kern;
        }

        //##############################################################################
        //#                               gauss                                        #
        //##############################################################################
        static MatrixLA gauss(MatrixLA x, double std)
        {
            Console.WriteLine("enter gauss");

            MatrixLA result = MatrixLA.Build.Dense(x.RowCount, x.ColumnCount);
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
            sizz = 2 * (int)Math.Ceiling(4.6 * stdd) + 1;
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

            double[][,] blurred_imgs = new double[maxscale + 1][,];
            double stdd;
            int sizz;

            //tuple to hold stdd, sizz
            Tuple<double, int> values_for_gauss;
            MatrixLA kern, c1mimg, cres;
            blurred_imgs[0] = new double[mimg.GetLength(0), mimg.GetLength(1)];

            for (int scale = 1; scale <= maxscale; scale++)
            {
                if (scale < 3)
                {
                    // Image is unblurred:
                    blurred_imgs[scale] = new double[mimg.GetLength(0), mimg.GetLength(1)];
                    for (int i = 0; i < mimg.GetLength(0); i++)
                    {
                        for (int j = 0; j < mimg.GetLength(1); j++)
                        {
                            blurred_imgs[scale][i, j] = mimg[i, j];
                        }
                    }

                }
                else
                {
                    // Set values for generating Gaussian filter at given scale:
                    values_for_gauss = setvalues(scale);
                    stdd = values_for_gauss.Item1;
                    sizz = values_for_gauss.Item2;
                    kern = d2gauss(sizz, stdd, 1, 1, 0, 1 / (stdd * Math.Sqrt(2 * Math.PI)));
                    MatrixLA mimg_matrix = MatrixLA.Build.DenseOfArray(mimg);
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
        static MatrixLA RotateMatrix90(MatrixLA oldMatrix)
        {
            MatrixLA newMatrix = MatrixLA.Build.Dense(oldMatrix.ColumnCount, oldMatrix.RowCount);
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
        static MatrixLA RotateMatrix90MultiTimes(MatrixLA matrix, int times)
        {
            for (int i = 0; i < times; i++)
            {
                MatrixLA matrix2 = RotateMatrix90(matrix);
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
            MatrixLA g1mag = Matrix.Build.Dense(g1x.GetLength(0), g1x.GetLength(1));
            //initial g1dir with 4
            MatrixLA g1dir = Matrix.Build.Dense(g1x.GetLength(0), g1x.GetLength(1), 4.0);
            for (int i = 0; i < g1x.GetLength(0); i++)
            {
                for (int j = 0; j < g1x.GetLength(1); j++)
                {
                    if ((Math.Abs(g1x[i, j]) > 0.00000000001) && (Math.Abs(g1y[i, j]) > 0.00000000001))
                    {
                        g1dir[i, j] = Math.Atan2(-g1y[i, j], g1x[i, j]);
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
        static double[][,] g1scale(double[,] g1mag1, double[,] g1dir1, double[,] g1mag2, double[,] g1dir2, double[,] g1sc1, int scale, double noise, int b_est)
        {
            Console.WriteLine("enter g1scale");
            double[][,] result = new double[3][,];
            int krad;
            double[] norms12 = { 0.765, 0.199, 0.0499, 0.0125, 0.00312, 0.00078 };
            double thresh = 5.6 * noise * norms12[scale - 1];
            if (scale < 3 || b_est == 1)
            {
                krad = 1;
            }
            else
            {
                krad = (int)Math.Ceiling(4.6 * Math.Sqrt(Math.Pow(2, 2 * (scale - 2)) - 1));
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

                //Console.WriteLine("g1mag2 scale = 1");
                //printMatrix(g1mag2, 299, 310, 299, 310);
                //Console.WriteLine("g1sc1 scale = 1");
                //printMatrix(g1sc1, 299, 310, 299, 310);

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
                    double[,] scaleMatrix = new double[g1sc1.GetLength(0) - 2 * krad, g1sc1.GetLength(1) - 2 * krad];
                    for (int i = 0; i < g1sc1.GetLength(0) - 2 * krad; i++)
                    {
                        for (int j = 0; j < g1sc1.GetLength(1) - 2 * krad; j++)
                        {
                            scaleMatrix[i, j] = g1sc1[krad + i, krad + j];
                        }
                    }

                    //magMatrix : mmat
                    double[,] magMatrix = new double[g1mag2.GetLength(0) - 2 * krad, g1mag2.GetLength(1) - 2 * krad];
                    for (int i = 0; i < g1mag2.GetLength(0) - 2 * krad; i++)
                    {
                        for (int j = 0; j < g1mag2.GetLength(1) - 2 * krad; j++)
                        {
                            magMatrix[i, j] = g1mag2[krad + i, krad + j];
                        }
                    }


                    for (int i = 0; i < scaleMatrix.GetLength(0); i++)
                    {
                        for (int j = 0; j < scaleMatrix.GetLength(1); j++)
                        {
                            //f = find((smat==0) & (mmat>=thresh));
                            if (Math.Abs(scaleMatrix[i, j]) < 0.00000000001 && (magMatrix[i, j] >= thresh))
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
                    Console.WriteLine("g1scale: matrix_i or matrix_j is empty");
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
            //^^^^^^^^^^^^^^^^^^^^^^^^^
            //^    initialize
            //^^^^^^^^^^^^^^^^^^^^^^^^^
            double[,] g1mag1 = Matrix.Build.Dense(gauss_a[0].GetLength(0), gauss_a[0].GetLength(1)).ToArray();
            double[,] g1dir1 = Matrix.Build.Dense(gauss_a[0].GetLength(0), gauss_a[0].GetLength(1)).ToArray();
            double[,] g1sc1 = Matrix.Build.Dense(gauss_a[0].GetLength(0), gauss_a[0].GetLength(1)).ToArray();

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

                MatrixLA mimg = Matrix.Build.DenseOfArray(gauss_a[scale]);


                //^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                //^     Compute response of first derivative Gaussian filter
                //^     to the blurred image in an arbitrary direction:
                //^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

                //kern1
                string[] kern1_str = System.IO.File.ReadAllLines("/Users/leo/Projects/DetectEdges/DetectEdges/filters/gy" + g1scaleval + fm2);
                MatrixLA kern1 = Matrix.Build.DenseOfArray(read_gy(kern1_str));
                MatrixLA rc1 = convolve_2(mimg, kern1, conv_type);

                //kern2
                string[] kern2_str = System.IO.File.ReadAllLines("/Users/leo/Projects/DetectEdges/DetectEdges/filters/g1x" + g1scaleval + fm2);
                MatrixLA kern2 = Matrix.Build.DenseOfArray(read_gx(kern2_str));
                MatrixLA rc2 = convolve_2(rc1, kern2, conv_type);

                //kern3
                string[] kern3_str = System.IO.File.ReadAllLines("/Users/leo/Projects/DetectEdges/DetectEdges/filters/gx" + g1scaleval + fm2);
                MatrixLA kern3 = Matrix.Build.DenseOfArray(read_gx(kern3_str));
                MatrixLA rc3 = convolve_2(mimg, kern3, conv_type);

                //kern4
                string[] kern4_str = System.IO.File.ReadAllLines("/Users/leo/Projects/DetectEdges/DetectEdges/filters/g1y" + g1scaleval + fm2);
                MatrixLA kern4 = Matrix.Build.DenseOfArray(read_gy(kern4_str));
                MatrixLA rc4 = convolve_2(rc3, kern4, conv_type);

                //^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                //^     Calculate magnitude and direction of the gradient:
                //^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                //Console.WriteLine("rc2");
                //printMatrix(rc2, 199, 210, 199, 210);
                //Console.WriteLine("rc4");
                //printMatrix(rc4, 199, 210, 199, 210);
                double[][,] g1steer_result = g1steer(rc2.ToArray(), rc4.ToArray());
                double[,] m2 = g1steer_result[0];
                double[,] d2 = g1steer_result[1];

                //^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                //^     Augment multi-scale Gaussian Gradient maps:
                //^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                g1scale_result = g1scale(g1mag1, g1dir1, m2, d2, g1sc1, scale, noise, 0);
                //Console.WriteLine("m2 scale = 1");
                //printMatrix(m2, 199, 210, 199, 210);
                g1mag1 = g1scale_result[0];
                g1dir1 = g1scale_result[1];
                g1sc1 = g1scale_result[2];

                //Console.WriteLine("g1mag scale = {0}", scale);
                //printMatrix(g1mag1, 399, 410, 399, 410);

                //Console.WriteLine("g1dir scale = {0}", scale);
                //printMatrix(g1dir1, 399, 410, 399, 410);

                //Console.WriteLine("g1sc scale = {0}", scale);
                //printMatrix(g1sc1, 399, 410, 399, 410);

            }
            return g1scale_result;
        }

        //##############################################################################
        //#                               g2steer
        //#     g2steer - Computes the second Gaussian derivative of the luminance function
        //# in specified direction(normally the Gradient direction). The three input
        //# basis function are used to steer the derivative.The units of the direction
        //# map are radians and the range is between -pi and pi.A value of -4 indicate
        //# that no direction was measurable.The 2nd derivative is taken only for valid
        //# directions.
        //# 
        //#       
        //#        Input:  g2x - Response map for 1st G2 basis function
        //#                g2y - Response map for 2nd G2 basis function
        //#                g2xy - Response map for 3rd G2 basis function
        //#                g1dir - Luminance gradient direction map
        //#
        //#        Output: g2 - Second derivative response map
        //#                
        //##############################################################################

        static double[,] g2steer(double[,] g2x, double[,] g2y, double[,] g2xy, double[,] g1dir)
        {
            double[,] g2 = new double[g2x.GetLength(0), g2x.GetLength(1)];
            double cdir, sdir;

            for (int i = 0; i < g2x.GetLength(0); i++)
            {
                for (int j = 0; j < g2x.GetLength(1); j++)
                {
                    if ((Math.Abs(g1dir[i, j] - 4.0) > 0.00000000001) && Math.Abs(g2x[i, j]) > 0.00000000001 && Math.Abs(g2xy[i, j]) > 0.00000000001 && Math.Abs(g2y[i, j]) > 0.00000000001)
                    {
                        cdir = Math.Cos(2 * g1dir[i, j]);
                        sdir = Math.Sin(2 * g1dir[i, j]);

                        g2[i, j] = 0.5 * (1 + cdir) * g2x[i, j] - sdir * g2xy[i, j] + 0.5 * (1 - cdir) * g2y[i, j];
                    }
                }
            }

            Console.WriteLine("equalToZero = {0}", Math.Abs(g2xy[403, 407]) > Double.Epsilon);

            return g2;
        }

        //##############################################################################
        //#
        //# g2scale - Augments multi-scale Gaussian directional 2nd derivative maps
        //# with significant estimates at a new scale. Pixels for which the magnitude
        //# of the 2nd derivative is under threshold in the multi-scale map 
        //# (2nd derivative input 1) but over threshold at the new scale(2nd derivative
        //# input 2) are updated with the 2nd derivative magnitude and scale value of the
        //# new scale.
        //#
        //# Input:  g2mag1   - Multi-scale Gaussian directional 2nd derivative image
        //#         g2mag2   - Gaussian directional 2nd derivative image at new scale
        //#         g1scale1 - Multi-scale scale map
        //#         noise    - Estimated sensor noise
        //#         b_est    - Estimate derivatives near boundaries?
        //#
        //# Output: g2mag   - Integrated multi-scale directional 2nd derivative map
        //#         g2scale - Integrated multi-scale 2nd derivative scale map
        //#
        //##############################################################################

        static double[][,] g2scale(double[,] g2mag1, double[,] g2mag2, double[,] g2sc1, int scale, int noise, int b_est)
        {
            double[][,] result = new double[2][,];
            double[] norms12 = { 1.873, 0.2443, 0.0306, 0.003871, 0.00047715, 0.0000596, 0.000007455 };
            double thresh = 5.2 * noise * norms12[scale - 1];
            int krad;

            if ((scale < 3) || b_est == 1)
            {
                krad = 1;
            }
            else
            {
                krad = (int)Math.Ceiling(4.6 * Math.Sqrt(Math.Pow(2, 2 * (scale - 2)) - 1.0));
            }

            if (scale == 1)
            {
                g2mag1 = Matrix.Build.Dense(g2mag2.GetLength(0), g2mag2.GetLength(1)).ToArray();
                g2sc1 = Matrix.Build.Dense(g2mag2.GetLength(0), g2mag2.GetLength(1)).ToArray();

                for (int i = 0; i < g2mag1.GetLength(0); i++)
                {
                    for (int j = 0; j < g2mag1.GetLength(1); j++)
                    {
                        if (Math.Abs(g2mag2[i, j]) >= thresh)
                        {
                            g2mag1[i, j] = g2mag2[i, j];
                            g2sc1[i, j] = scale;
                        }
                    }
                }
                result[0] = g2mag1;
                result[1] = g2sc1;
                return result;
            }
            else
            {
                int[] sz = new int[2];
                sz[0] = g2mag2.GetLength(0);
                sz[1] = g2mag2.GetLength(1);
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

                    //magMatrix1 : magmat1
                    double[,] magMatrix1 = new double[g2mag1.GetLength(0) - 2 * krad, g2mag1.GetLength(1) - 2 * krad];
                    for (int i = 0; i < g2mag1.GetLength(0) - 2 * krad; i++)
                    {
                        for (int j = 0; j < g2mag1.GetLength(1) - 2 * krad; j++)
                        {
                            magMatrix1[i, j] = g2mag1[krad + i, krad + j];
                        }
                    }

                    //magMatrix2 : magmat2
                    double[,] magMatrix2 = new double[g2mag2.GetLength(0) - 2 * krad, g2mag2.GetLength(1) - 2 * krad];
                    for (int i = 0; i < g2mag2.GetLength(0) - 2 * krad; i++)
                    {
                        for (int j = 0; j < g2mag2.GetLength(1) - 2 * krad; j++)
                        {
                            magMatrix2[i, j] = g2mag2[krad + i, krad + j];
                        }
                    }

                    for (int i = 0; i < magMatrix1.GetLength(0); i++)
                    {
                        for (int j = 0; j < magMatrix1.GetLength(1); j++)
                        {
                            //f = find(abs(magmat1) == 0 & abs(magmat2) >= thresh);
                            if (Math.Abs(magMatrix1[i, j]) < 0.00000000001 && Math.Abs(magMatrix2[i, j]) >= thresh)
                            {
                                //g1mag1(K(f)) = g1mag2(K(f));
                                //g1dir1(K(f)) = g1dir2(K(f));
                                //g1sc1(K(f)) = scale;
                                g2mag1[matrix_i[i, j], matrix_j[i, j]] = g2mag2[matrix_i[i, j], matrix_j[i, j]];
                                g2sc1[matrix_i[i, j], matrix_j[i, j]] = scale;
                            }
                        }
                    }
                }
                else
                {
                    Console.WriteLine("g2scale: matrix_i or matrix_j is empty");
                }
                result[0] = g2mag1;
                result[1] = g2sc1;

                Console.WriteLine("g2mag1 scale={0}", scale);
                printMatrix(g2mag1, 399, 410, 399, 410);

                return result;
            }
        }

        //##############################################################################
        //#    derivative2nd(g1dir,maxscale,noise,gauss_a,conv_type,fpath)
        //##############################################################################

        static double[][,] derivative2nd(double[,] g1dir, int maxscale, int noise, double[][,] gauss_a, int conv_type)
        {
            double[][,] g2scale_result = new double[2][,];
            double[][,] result = new double[3][,];
            double[,] g2;
            String g2scaleval, fm2 = ".ascii";

            //^^^^^^^^^^^^^^^^^^^^^^^^^
            //^    initialize
            //^^^^^^^^^^^^^^^^^^^^^^^^^
            double[,] g2mag1 = Matrix.Build.Dense(gauss_a[0].GetLength(0), gauss_a[0].GetLength(1)).ToArray();
            double[,] g2sc1 = Matrix.Build.Dense(gauss_a[0].GetLength(0), gauss_a[0].GetLength(1)).ToArray();
            int nrows = g1dir.GetLength(0);
            double[,] g2all = new double[maxscale * nrows, g1dir.GetLength(1)];

            for (int scale = 1; scale <= maxscale; scale++)
            {
                if (scale == 1)
                {
                    g2scaleval = "05";
                }
                else
                {
                    g2scaleval = "1";
                }

                MatrixLA mimg = Matrix.Build.DenseOfArray(gauss_a[scale]);

                //^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                //^     Compute response of second derivative Gaussian filter
                //^     to the blurred image in an arbitrary direction:
                //^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

                //kern1
                string[] kern1_str = System.IO.File.ReadAllLines("/Users/leo/Projects/DetectEdges/DetectEdges/filters/gx" + g2scaleval + fm2);
                MatrixLA kern1 = Matrix.Build.DenseOfArray(read_gx(kern1_str));
                MatrixLA rc1 = convolve_2(mimg, kern1, conv_type);

                //kern2
                string[] kern2_str = System.IO.File.ReadAllLines("/Users/leo/Projects/DetectEdges/DetectEdges/filters/g2y" + g2scaleval + fm2);
                MatrixLA kern2 = Matrix.Build.DenseOfArray(read_gy(kern2_str));
                MatrixLA rc2 = convolve_2(rc1, kern2, conv_type);

                //kern3
                string[] kern3_str = System.IO.File.ReadAllLines("/Users/leo/Projects/DetectEdges/DetectEdges/filters/gy" + g2scaleval + fm2);
                MatrixLA kern3 = Matrix.Build.DenseOfArray(read_gy(kern3_str));
                MatrixLA rc3 = convolve_2(mimg, kern3, conv_type);

                //kern4
                string[] kern4_str = System.IO.File.ReadAllLines("/Users/leo/Projects/DetectEdges/DetectEdges/filters/g2x" + g2scaleval + fm2);
                MatrixLA kern4 = Matrix.Build.DenseOfArray(read_gx(kern4_str));
                MatrixLA rc4 = convolve_2(rc3, kern4, conv_type);

                //kern5
                string[] kern5_str = System.IO.File.ReadAllLines("/Users/leo/Projects/DetectEdges/DetectEdges/filters/g1x" + g2scaleval + fm2);
                MatrixLA kern5 = Matrix.Build.DenseOfArray(read_gx(kern5_str));
                MatrixLA rc5 = convolve_2(mimg, kern5, conv_type);

                //kern6
                string[] kern6_str = System.IO.File.ReadAllLines("/Users/leo/Projects/DetectEdges/DetectEdges/filters/g1y" + g2scaleval + fm2);
                MatrixLA kern6 = Matrix.Build.DenseOfArray(read_gy(kern6_str));
                MatrixLA rc6 = convolve_2(rc5, kern6, conv_type);

                //^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                //^     Calculate the 2nd Gaussian derivative:
                //^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

                g2 = g2steer(rc4.ToArray(), rc2.ToArray(), rc6.ToArray(), g1dir);
                for (int i = 0; i < g1dir.GetLength(0); i++)
                {
                    for (int j = 0; j < g1dir.GetLength(1); j++)
                    {
                        g2all[(scale - 1) * g1dir.GetLength(0) + i, j] = g2[i, j];
                    }
                }

                //^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                //^     Augment multi-scale Gaussian directional 2nd derivative maps:
                //^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

                g2scale_result = g2scale(g2mag1, g2, g2sc1, scale, noise, 0);
                g2mag1 = g2scale_result[0];
                g2sc1 = g2scale_result[1];
            }
            result[0] = g2scale_result[0];
            result[1] = g2scale_result[1];
            result[2] = g2all;
            return result;
        }

    }
}
