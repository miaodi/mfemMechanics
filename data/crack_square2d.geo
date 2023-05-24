
dt = 1e-3;

Point(1) = {-5e-3, -5e-3, 0, dt};
Point(2) = {5e-3, -5e-3, 0, dt};
Point(3) = {5e-3, 5e-3, 0, dt};
Point(4) = {-5e-3, 5e-3, 0, dt};
Point(5) = {-5e-3, 5e-5, 0, dt};
Point(6) = {0, 0, 0, dt};
Point(7) = {-5e-3, -5e-5, 0, dt};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 5};
Line(5) = {5, 6};
Line(6) = {6, 7};
Line(7) = {7, 1};

Curve Loop(1) = {1, 2, 3, 4, 5, 6, 7};
Plane Surface(1) = {1};

Physical Surface(1) = {1};

Mesh 2;
Mesh.MshFileVersion = 2.2;
