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


Különböző fényviszonyoknál tévedések előfordulnak
!! Amikor a camera night modeba kapcsol az összes szektort aktívnak érzékeli !!