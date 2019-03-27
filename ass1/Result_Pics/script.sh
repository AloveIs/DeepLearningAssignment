#!/bin/sh


for f in *.pdf
do
	pdfcrop --margins '3 3 3 3' --clip $f $f
done
