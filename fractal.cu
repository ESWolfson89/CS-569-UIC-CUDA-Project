#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include "GL/glut.h"
#include <math.h>
#include <iostream>
#include <stdio.h>
#include <time.h>

using namespace std;

static const int WID = 768;
static const int HGT = 768;
static const unsigned MaxIterations = 500;

clock_t clock_timer_var;

#define BLOCK_SIZE 16

int main_window;

// not used
struct color
{
	GLdouble r, g, b;
};

__global__ void drawMandelbrotGPU(double MaxRe, double MinRe, double MaxIm, double MinIm, color *cell_array);

double cell_arrayr[WID * HGT];

int iteration = 1;
bool GPU_mode = false;
bool update_screen = true;
double MinRe = -2.0;
double MaxRe = 1.0;
double MinIm = -1.5;
double MaxIm = 1.5;
double by = 1.0;
double bx = 1.0;
double ey = 1.0;
double ex = 1.0;
double cy = 1.0, cx = 1.0;
bool draw = true;

void resetZoomParameters()
{
   MinRe	=	-2.0;
   MaxRe	=	1.0;
   MinIm	=	-1.5;
   MaxIm	=	1.5;
}

void init(int *argc_ptr, char** argv)
{
   glClearColor(0.0,0.0,0.0,0.0);       // background color is black
   glColor3f(0.0f, 0.0f, 0.0f);         // drawing color is black (initially) 
   glPointSize(1.0);                          // a 'dot' is 1 by 1 pixel
   glMatrixMode(GL_PROJECTION);            // set "camera shape"
   glLoadIdentity();
   gluOrtho2D(0.0, (GLdouble)WID, 0.0, (GLdouble)HGT);
}

void drawSavedMap()
{
	glBegin(GL_POINTS);
	color c;
	for (int i = 0; i < WID; ++i)
	{
		for (int j = 0; j < HGT; ++j)
		{
			c.r = cell_arrayr[i*WID + j];	
			glColor3f(c.r,c.r/4,0.0f);
			glVertex2f(i,j);
		}
	}
	glEnd();
}	

void printInfo()
{
    cout << "Last redraw/compute:" << endl;
    cout << "size of double: " << sizeof(double) << " size of long double: " << sizeof(long double) << endl;

    if (GPU_mode == false)
        cout << "********CPU MODE********";
    else
        cout << "********GPU MODE********";

    cout << endl << "Max iterations: " << MaxIterations << " BLOCK_SIZE: " << BLOCK_SIZE;

    cout << endl << "Min x (real axis): " << MinRe << endl << "Max x (real axis): " << MaxRe;
    cout << endl << "Min y (complex axis): " << MinIm << endl << "Max y (complex axis): " << MaxIm;
    cout << endl << "Time to calculate Mandelbrot & draw screen: " << (clock() - clock_timer_var ) / (double) CLOCKS_PER_SEC << "s";
    cout << endl;
}

void drawBox()
{
    glColor3f(1.0f,1.0f,1.0f);
    glBegin(GL_LINES);
    glVertex2f(bx,HGT-by);
    glVertex2f(cx,HGT-by);
    glEnd();
    glBegin(GL_LINES);
    glVertex2f(bx,HGT-by);
    glVertex2f(bx,cy);
    glEnd();
    glBegin(GL_LINES);
    glVertex2f(cx,HGT-by);
    glVertex2f(cx,cy);
    glEnd();
    glBegin(GL_LINES);
    glVertex2f(bx,cy);
    glVertex2f(cx,cy);
    glEnd();
}

__global__ void drawMandelbrotGPU(double MaxRe, double MinRe, double MaxIm, double MinIm, double *cell_arrayr)
{
  // r stands for row, c for column
  int r = blockIdx.y * blockDim.y + threadIdx.y;
  int c = blockIdx.x * blockDim.x + threadIdx.x; 
  int index = (c * HGT) + r;

  if(c >= WID || r >= HGT) 
	 return;

  double Re_factor	= (MaxRe - MinRe) / WID;
  double Im_factor	= (MaxIm - MinIm) / HGT;  
  double c_im, c_re, Z_re, Z_im, Z_re2, Z_im2;
  double loop_iters_n;

  c_im = MaxIm - r * Im_factor;
  c_re = MinRe + c * Re_factor;
  Z_re = c_re;
  Z_im = c_im;
  loop_iters_n = 0;

  for (unsigned n = 0; n < MaxIterations; ++n ) 
  {
      Z_re2 = Z_re * Z_re;
      Z_im2 = Z_im * Z_im;
      if ( Z_re2 + Z_im2 > 4 )
          break;
      Z_im = 2 * Z_re * Z_im + c_im;
      Z_re = Z_re2 - Z_im2 + c_re;
      loop_iters_n = (double)n;
  }

  loop_iters_n /= MaxIterations;
  cell_arrayr[index] = loop_iters_n;
}

void drawMandelbrotCPU()
{
  glClear(GL_COLOR_BUFFER_BIT);
  double Re_factor	=	(MaxRe - MinRe) / WID;
  double Im_factor	=	(MaxIm - MinIm) / HGT;  
  double c_im, c_re, Z_re, Z_im, Z_re2, Z_im2;
  double loop_iters_n;
  
  glBegin(GL_POINTS);

  for ( unsigned y = 0; y < HGT; ++y ) 
  {
	  c_im = MaxIm - y * Im_factor;
	  for ( unsigned x = 0; x < WID; ++x ) 
	  {
		  c_re = MinRe + x * Re_factor;
		  Z_re = c_re;
		  Z_im = c_im;
		  color col;
		  loop_iters_n = 0;
		  for ( unsigned n = 0; n < MaxIterations; ++n ) 
		  {	
			  Z_re2 = Z_re * Z_re;
			  Z_im2 = Z_im * Z_im;
			  if ( Z_re2 + Z_im2 > 4 )
				  break;
			  Z_im = 2 * Z_re * Z_im + c_im;
			  Z_re = Z_re2 - Z_im2 + c_re;
			  loop_iters_n = (double)n;
		  }
		  loop_iters_n /= MaxIterations;
		  col.r = loop_iters_n;
		  glColor3f(col.r,col.r/4,0.0f);
		  glVertex2f(x,y);
		  cell_arrayr[(x*WID)+y] = loop_iters_n;
	  }
  }

  glEnd();
}

void reDisplay(void)
{
  if (update_screen)
  {
  	if (draw == true)
  	{
	  clock_timer_var = clock();

	  if (GPU_mode == false)
  		drawMandelbrotCPU();
	  else
	  {
		double *array_temp;
		size_t size = sizeof(double) * WID * HGT;
		cudaMalloc(&array_temp,size);
		cudaMemcpy(array_temp,cell_arrayr,size,cudaMemcpyHostToDevice);
		dim3 b_size(BLOCK_SIZE, BLOCK_SIZE);
		dim3 grid(WID / b_size.x, HGT / b_size.y);	
		drawMandelbrotGPU<<<grid,b_size>>>(MaxRe, MinRe, MaxIm, MinIm, array_temp);
		cudaDeviceSynchronize();
		cudaMemcpy(cell_arrayr,array_temp,size,cudaMemcpyDeviceToHost);
		cudaFree(array_temp);
		glClear(GL_COLOR_BUFFER_BIT);
		drawSavedMap();
	  }
          printInfo();
          update_screen = false;
  	}
  	else
  	{
  		drawSavedMap();
  		drawBox();
  	}
  	glFlush();
  }	
}


void keyPress(unsigned char k, int x, int y)
{
	if (k == 'g')
		GPU_mode = true;
	if (k == 'c')
		GPU_mode = false;
	if (k == 'q')
		exit(0);
	if (k == '1')
	{
		MinRe = -0.480469;
		MaxRe = 0.179688;
		MinIm = -0.402344;
		MaxIm = 0.371094;
		draw = true;
		update_screen = true;
	}
	if (k == '2')
	{
		MinRe = -1.51172;
		MaxRe = -0.820313;
		MinIm = -1.18359;
		MaxIm = -0.675781;
		draw = true;
		update_screen = true;
	}
	if (k == '3')
	{
		MinRe = -0.806519;
		MaxRe = -0.776896;
		MinIm = -0.168213;
		MaxIm = -0.137777;
		draw = true;
		update_screen = true;
	}
	if (k == '4')
	{
		MinRe = -0.755904;
		MaxRe = -0.75399;
		MinIm = -0.0578789;
		MaxIm = -0.0563382;
		draw = true;
		update_screen = true;
	}
	if (k == '5')
	{
		MinRe = 0.269364;
		MaxRe = 0.269776;
		MinIm = -0.00462961;
		MaxIm = -0.00434589;
		draw = true;
		update_screen = true;
	}
	if (k == 'r')
	{
            resetZoomParameters();
            draw = true;		
            update_screen = true;
	}
}

void mouseMove(int x, int y)
{
	cx = x;
	cy = y;
  update_screen = true;
}

void mouseClick(int button, int state, int x, int y)
{
	if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN)
	{
		draw = false;
		by = cy = y;
		bx = cx = x;
	}
	else if (button == GLUT_LEFT_BUTTON && state == GLUT_UP)
	{
		draw = true;
		ey = y;
		ex = x;
		double tempMinRe = MinRe, tempMaxRe = MaxRe;
		double tempMinIm = MinIm, tempMaxIm = MaxIm;
		MinRe = ((min(bx,ex)/WID) * (tempMaxRe - tempMinRe)) + tempMinRe;
		MaxRe = ((max(bx,ex)/WID) * (tempMaxRe - tempMinRe)) + tempMinRe;
		MinIm = ((min(by,ey)/HGT) * (tempMaxIm - tempMinIm)) + tempMinIm;
		MaxIm = ((max(by,ey)/HGT) * (tempMaxIm - tempMinIm)) + tempMinIm;
		update_screen = true;
	}
}

void wrapDisplay(void) { reDisplay(); }

void wrapMouse(int b, int st, int x, int y)
{
	mouseClick(b,st,x,y);
}

void wrapMotion(int x, int y)
{
	mouseMove(x,HGT-y);
}

void wrapKey(unsigned char k, int x, int y)
{
	keyPress(k,x,(HGT-y));
}

void timer_func(int value)     
{     
	glutPostRedisplay();
	glutTimerFunc(1, timer_func, 0);
}


//<<<<<<<<<<<<<<<<<<<<<<<< main >>>>>>>>>>>>>>>>>>>>>>
int main(int argc, char* argv[])
{
      srand(time(NULL));
      
      glutInit(&argc, argv);        
      glutInitWindowSize(WID, HGT);
      glutInitWindowPosition(100, 100); 
      glutInitDisplayMode(GLUT_RGB); 
      
      main_window = glutCreateWindow( "Mandelbrot By Eric Wolfson");
      
      init(&argc, argv); // initialize the display    
      
      glutDisplayFunc(wrapDisplay);   // register Redraw function
      glutMotionFunc(wrapMotion);
      glutMouseFunc(wrapMouse);
      glutKeyboardFunc(wrapKey);
      glutTimerFunc(1,timer_func, 0);
    
      glutMainLoop(); // perpetual loop
      cudaDeviceReset();
      
      return 0;
} 
