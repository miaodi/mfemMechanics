
dt = 1e-4;

Point(1) = {-5e-4, -5e-4, 0, dt};
Point(2) = {5e-4, -5e-4, 0, dt};
Point(3) = {5e-4, 5e-4, 0, dt};
Point(4) = {-5e-4, 5e-4, 0, dt};
Point(5) = {-5e-4, 5e-6, 0, dt};
Point(6) = {0, 0, 0, dt};
Point(7) = {-5e-4, -5e-6, 0, dt};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 5};
Line(5) = {5, 6};
Line(6) = {6, 7};
Line(7) = {7, 1};

Curve Loop(1) = {1, 2, 3, 4, 5, 6, 7};
Plane Surface(1) = {1};

Extrude {0, 0, 1e-3}{ Surface {1}; }
Physical Volume(1) = {1};

Physical Surface("bot", 11) = {19};
Physical Surface("top", 12) = {27};
Physical Surface("right", 13) = {23};
Physical Surface("left top", 14) = {31};
Physical Surface("left bot", 15) = {43};
Physical Surface("crack_bot", 16) = {39};
Physical Surface("crack_top", 17) = {35};


Mesh 3;
Mesh.Algorithm = 8; // Delaunay for quads
Mesh.MshFileVersion = 2.2;
