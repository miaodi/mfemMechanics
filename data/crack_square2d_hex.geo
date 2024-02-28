// Gmsh project created on Tue Jan 17 13:21:26 2023
// 0 for triangles, 1 for quads
tri_or_quad = 1;
dt = .5e-4;

Point(1)   ={-5e-4 ,-5e-4      ,0,dt};
Point(2)   ={-5e-4 ,-5e-6      ,0,dt};
Point(3)   ={-5e-4 ,5e-6       ,0,dt};
Point(4)   ={-5e-4 ,5e-4       ,0,dt};
Point(5)   ={5e-4  ,-5e-4      ,0,dt};
Point(6)   ={5e-4  ,5e-4       ,0,dt};
Point(7)   ={0     ,0          ,0,dt};
Point(8)   ={5e-4  ,0          ,0,dt};
Point(9)   ={0     ,5e-4       ,0,dt};
Point(10)  ={0     ,-5e-4      ,0,dt};


//Characteristic Length {:} = 1e-4;

Line(1)  ={1 ,10};
Line(2)  ={5 ,8};
Line(3)  ={6 ,9};
Line(4)  ={4 ,3};
Line(5)  ={3 ,7};
Line(6)  ={7 ,2};
Line(7)  ={2 ,1};

Line(8)  ={10,7};
Line(9)  ={7 ,9};
Line(10) ={7 ,8};
Line(11) ={9 ,4};
Line(12) ={10,5};
Line(13) ={8 ,6};

Curve Loop(1) = {1, 8, 6, 7};
Plane Surface(1) = {1};
Curve Loop(2) = {12, 2, -10, -8};
Plane Surface(2) = {2};
Curve Loop(3) = {10, 13, 3, -9};
Plane Surface(3) = {3};
Curve Loop(4) = {5, 9, 11, 4};
Plane Surface(4) = {4};
Transfinite Surface {1, 2, 3, 4};

If (tri_or_quad == 1)
   Recombine Surface {1, 2, 3, 4};
EndIf

Physical Surface(1) = {1, 2, 3, 4};

Physical Curve("bot", 11) = {1, 12};
Physical Curve("top", 12) = {3, 11};
Physical Curve("right", 13) = {2, 13};
Physical Curve("left top", 14) = {4};
Physical Curve("left bot", 15) = {7};

Mesh.MshFileVersion = 2.2;
Mesh 2;
