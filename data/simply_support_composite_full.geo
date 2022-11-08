        // Inputs
	a = .254; //m
	h = 0.001;
	dt = .01;
 
        // Geometry
	Point(1) = {0, 0, 0};
	Point(2) = {a, 0, 0};
	Point(3) = {a, a, 0};
	Point(4) = {0, a, 0};

	
	Characteristic Length {:} = dt;
	Line(1) = {1, 2};				// bottom line
	Line(2) = {2, 3};				// right line
	Line(3) = {3, 4};				// top line
	Line(4) = {4, 1};				// left line
	Line Loop(5) = {1, 2, 3, 4}; 	
	Plane Surface(6) = {5};
 
    //Transfinite surface:
	Transfinite Surface {6};
	Recombine Surface {6};
	surfaceVector[] = Extrude {0, 0, h} {
	 Surface{6};
	 Layers{1};
	 Recombine;
	};

	Physical Surface("top", 25) = {28};
	Physical Surface("bottom", 26) = {6};
	Physical Surface("left", 21) = {15};
	Physical Surface("right", 22) = {23};
	Physical Surface("front", 23) = {19};
	Physical Surface("back", 24) = {27};

	Physical Volume("body", 31) = {1};

	Mesh.MshFileVersion = 2.2;
	Mesh 3;
