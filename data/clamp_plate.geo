        // Inputs
	squareSide = 200; //m
	meshThickness = squareSide / 10; 
	gridsize = squareSide / 20;
 
        // Geometry
	Point(1) = {-squareSide/2, -squareSide/2, 0, gridsize};
	Point(2) = {squareSide/2, -squareSide/2, 0, gridsize};
	Point(3) = {squareSide/2, squareSide/2, 0, gridsize};
	Point(4) = {-squareSide/2, squareSide/2, 0, gridsize};
	Line(1) = {1, 2};				// bottom line
	Line(2) = {2, 3};				// right line
	Line(3) = {3, 4};				// top line
	Line(4) = {4, 1};				// left line
	Line Loop(5) = {1, 2, 3, 4}; 	
	Plane Surface(6) = {5};
 
        //Transfinite surface:
	Transfinite Surface {6};
	Recombine Surface {6};
 
	surfaceVector[] = Extrude {0, 0, meshThickness} {
	 Surface{6};
	 Layers{1};
	 Recombine;
	};
Mesh.MshFileVersion = 2.2;
Mesh 3;
