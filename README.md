Szükséges, hogy legyen telepítve: python3, opencv-python, numpy

detection.py 
detect_active_sectors(img) függvény visszad k arrayt, ha k[i] == 1 akkor ülnek az adott padsor szektorban, ha k[i] == 0 akkor nem

padsor szektorok:

        +-------------------------+
        |           OLTÁR         |
        +-------------------------+

--------------------    --------------------
-----elso-----------    ------negyedik------
--------------------    --------------------

--------------------    --------------------
-----masodik--------    ------otodik--------
--------------------    --------------------

--------------------    --------------------
-----harmadik-------    ------hatodik-------
--------------------    --------------------
--------------------    --------------------
--------------------    --------------------


Esetleges tévedések előfordulnak.
A detection.py futttatása elég a detektáláshoz, a switch.py-t kell módosítani amennyiben mást szeretnénk kapcsolni.

A sect1.jpg, sect2.jpg, ... sect6.jpg-nek egy mappában kell lennie a detection.py-al! 

!! Amikor a camera night modeba kapcsol nem érzékel semmit !!

Kódot írta: Miszori Gergő, Urbán Szabolcs, Bővíz Dániel