There are many scripts here to calculate errors and plot things in a specific way. They are there for reference for how the errors in the paper are calculated. 

----------------------------------------------------------download the dataset----------------------------------------------------------

download and extract the `bilateral_free_space_sep_27` folder from [link](https://vanderbilt365-my.sharepoint.com/personal/hao_yang_vanderbilt_edu/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fhao%5Fyang%5Fvanderbilt%5Fedu%2FDocuments%2F%5FMAPLE%2Dresearch%2FForce%2DEstimation%2FPenn&fromShare=true&ga=1) into `../`

----------------------------------------------------------train.py----------------------------------------------------------

This is the base script to train each network. There are variations on this for training each of the cases listed in the ISMR 2021 paper. It takes two arguemnts, the first is which data to load (choice between 'free_space' and 'trocar'). The second is a boolean for whether to use the RNN or not. It assumes the path to data is at '../data/csv/< 'train', 'val' >/< 'free_space', 'trocar' >'

	python train.py free_space 1
    

----------------------------------------------------------test.py----------------------------------------------------------

This is the base script to test each network. There are variations on this for testing each of the cases listed in the ISMR 2021 paper. It takes three arguemnts, the first is which experiment to load as a striong, the second is which network to use ('lstm' or not), and the third is whether to use the seal or base case. It assumes the path to data is at '../data/csv/test/< 'no_contact', 'with_contact' >/<exp>

	python test.py <exp> lstm seal