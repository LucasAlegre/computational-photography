hdr_image = makehdr({'office/office_1.jpg','office/office_2.jpg','office/office_3.jpg','office/office_4.jpg','office/office_5.jpg','office/office_6.jpg'});
imwrite(tonemap(hdr_image), "matlab_hdr_tonemap.png");

myhdr = load('hdr.mat').hdr;
imwrite(myhdr, "myhdr.t");