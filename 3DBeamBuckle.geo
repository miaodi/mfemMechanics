// Gmsh project created on Sun Jun 05 11:23:35 2022
// 0 for triangles, 1 for quads
tet_or_hex = 0;
b = 10 ;
h = 300 ;
L = 5000;
dt = 10;
Point(1) = {0 , -b/2, 0};
Point(2) = {0, b/2, 0};
Point(3) = {0, b/2, h};
Point(4) = {0, -b/2, h};
Characteristic Length {:} = dt;
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

Curve Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};
Transfinite Surface {1};


If (tet_or_hex == 1)
   Recombine Surface {1};
   out[] = Extrude {L, 0, 0} { Surface{1}; Layers{L/dt}; Recombine; };
Else
   out[] = Extrude {L, 0, 0} { Surface{1}; Layers{L/dt}; };
EndIf

Physical Surface("top", 27) = {21};
Physical Surface("bottom", 28) = {13};
Physical Surface("left", 29) = {1};
Physical Surface("right", 30) = {26};
Physical Surface("front", 32) = {25};

Physical Volume("body", 31) = {1};
Mesh.MshFileVersion = 2.2;
Mesh 3;
