{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.ensemble import RandomForestRegressor\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "df = pd.read_csv(\"https://raw.githubusercontent.com/dphi-official/Datasets/master/Loan_Data/loan_train.csv\", index_col=0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Loan_ID</th>\n",
       "      <th>Gender</th>\n",
       "      <th>Married</th>\n",
       "      <th>Dependents</th>\n",
       "      <th>Education</th>\n",
       "      <th>Self_Employed</th>\n",
       "      <th>ApplicantIncome</th>\n",
       "      <th>CoapplicantIncome</th>\n",
       "      <th>LoanAmount</th>\n",
       "      <th>Loan_Amount_Term</th>\n",
       "      <th>Credit_History</th>\n",
       "      <th>Property_Area</th>\n",
       "      <th>Loan_Status</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>LP002305</td>\n",
       "      <td>Female</td>\n",
       "      <td>No</td>\n",
       "      <td>0</td>\n",
       "      <td>Graduate</td>\n",
       "      <td>No</td>\n",
       "      <td>4547</td>\n",
       "      <td>0.0</td>\n",
       "      <td>115.0</td>\n",
       "      <td>360.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>Semiurban</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>LP001715</td>\n",
       "      <td>Male</td>\n",
       "      <td>Yes</td>\n",
       "      <td>3+</td>\n",
       "      <td>Not Graduate</td>\n",
       "      <td>Yes</td>\n",
       "      <td>5703</td>\n",
       "      <td>0.0</td>\n",
       "      <td>130.0</td>\n",
       "      <td>360.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>Rural</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>LP002086</td>\n",
       "      <td>Female</td>\n",
       "      <td>Yes</td>\n",
       "      <td>0</td>\n",
       "      <td>Graduate</td>\n",
       "      <td>No</td>\n",
       "      <td>4333</td>\n",
       "      <td>2451.0</td>\n",
       "      <td>110.0</td>\n",
       "      <td>360.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>Urban</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>LP001136</td>\n",
       "      <td>Male</td>\n",
       "      <td>Yes</td>\n",
       "      <td>0</td>\n",
       "      <td>Not Graduate</td>\n",
       "      <td>Yes</td>\n",
       "      <td>4695</td>\n",
       "      <td>0.0</td>\n",
       "      <td>96.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>1.0</td>\n",
       "      <td>Urban</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>LP002529</td>\n",
       "      <td>Male</td>\n",
       "      <td>Yes</td>\n",
       "      <td>2</td>\n",
       "      <td>Graduate</td>\n",
       "      <td>No</td>\n",
       "      <td>6700</td>\n",
       "      <td>1750.0</td>\n",
       "      <td>230.0</td>\n",
       "      <td>300.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>Semiurban</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "    Loan_ID  Gender Married Dependents     Education Self_Employed  \\\n",
       "0  LP002305  Female      No          0      Graduate            No   \n",
       "1  LP001715    Male     Yes         3+  Not Graduate           Yes   \n",
       "2  LP002086  Female     Yes          0      Graduate            No   \n",
       "3  LP001136    Male     Yes          0  Not Graduate           Yes   \n",
       "4  LP002529    Male     Yes          2      Graduate            No   \n",
       "\n",
       "   ApplicantIncome  CoapplicantIncome  LoanAmount  Loan_Amount_Term  \\\n",
       "0             4547                0.0       115.0             360.0   \n",
       "1             5703                0.0       130.0             360.0   \n",
       "2             4333             2451.0       110.0             360.0   \n",
       "3             4695                0.0        96.0               NaN   \n",
       "4             6700             1750.0       230.0             300.0   \n",
       "\n",
       "   Credit_History Property_Area  Loan_Status  \n",
       "0             1.0     Semiurban            1  \n",
       "1             1.0         Rural            1  \n",
       "2             1.0         Urban            0  \n",
       "3             1.0         Urban            1  \n",
       "4             1.0     Semiurban            1  "
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "Int64Index: 491 entries, 0 to 490\n",
      "Data columns (total 13 columns):\n",
      " #   Column             Non-Null Count  Dtype  \n",
      "---  ------             --------------  -----  \n",
      " 0   Loan_ID            491 non-null    object \n",
      " 1   Gender             481 non-null    object \n",
      " 2   Married            490 non-null    object \n",
      " 3   Dependents         482 non-null    object \n",
      " 4   Education          491 non-null    object \n",
      " 5   Self_Employed      462 non-null    object \n",
      " 6   ApplicantIncome    491 non-null    int64  \n",
      " 7   CoapplicantIncome  491 non-null    float64\n",
      " 8   LoanAmount         475 non-null    float64\n",
      " 9   Loan_Amount_Term   478 non-null    float64\n",
      " 10  Credit_History     448 non-null    float64\n",
      " 11  Property_Area      491 non-null    object \n",
      " 12  Loan_Status        491 non-null    int64  \n",
      "dtypes: float64(4), int64(2), object(7)\n",
      "memory usage: 53.7+ KB\n"
     ]
    }
   ],
   "source": [
    "df.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Loan_ID               0\n",
       "Gender               10\n",
       "Married               1\n",
       "Dependents            9\n",
       "Education             0\n",
       "Self_Employed        29\n",
       "ApplicantIncome       0\n",
       "CoapplicantIncome     0\n",
       "LoanAmount           16\n",
       "Loan_Amount_Term     13\n",
       "Credit_History       43\n",
       "Property_Area         0\n",
       "Loan_Status           0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:>"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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iIjI1DBNBcDNbFszu8rM3jCzf8zMzezuosclIiIiIiIiIiIiIg1P06IHUOJUYElgMNAbWKjY4YiIiIiIiIiIiIhIQzVNZYADRwELANMDBxU8FhERERERERERERFpwKapDHB3fyX72cyKHIqIiIiIiIiIiIiINHDTWga4iIiIiIiIiIiIiEhZTFMZ4OWw5ppretFjaMy6d+8OwJFHHlnoOBozfQbF02dQLL3/xdNnUDx9BsXTZ1Asvf/F02dQPH0GxdL7Xzx9BtOGV199tVJLPDSK+OMnn3zC0UcfzeWXX07Xrl2n1v9mqv+NVFwAXERERERERERERERqZ8yYMQwcOJA+ffoUPZSyUABcREREREREREREpBH4448/+O233+jfvz8DBgygf//+E/08aNAg3Mcnubdu3brAEdedAuAiIiIiIiJSdl27duXVV18tehiNWo8ePYoegoiITEP69u3LLrvswtixYye5jZkxxxxz0KVLF7p06cLiiy/OAgssUI+jLD8FwEVERERERKTsevToodq7BcrqH4uIiGQ6duzIBRdcQO/evcdle5dmf48ZM4aePXvSs2dPXnnlFZo2bcott9zCnHPOWfTwa00BcBEREREREREREZEKZ2Yst9xyLLfccowcOXKiAPjff/9Nz549+fHHH+nZsycAo0ePZsCAAQqAi4iIiIiIiIiIiMi0659//uGEE06gd+/eDB48uNptzIzZZ5+dpZZaig4dOtC5c2cWWmiheh5peU1TAXAz2xLYMv1zlvTflczs9vRzP3c/tp6HJSIiIiIiItLgqA57sVSDXUSmNc2aNaNLly60aNFiXPb3kCFDJtjG3enVqxeDBg1iwIABDB06lOHDh9OiRYuCRl1301QAHOgK7FFy37zpBvAroAC4iIiIiIiIyH9QHfbiqAa7iEyLWrVqxbHHThhazZdCKS2J8ssvv/DOO+/w888/07Vr12IGXQbTVADc3bsB3QoehoiIiIiIiIiIiEjFa968ObPMMguzzDLLRI998sknHH300QWMqrymqQC4iIiIiIiIiIiIiBRj+PDh4zLAv/zyy6KHUxYKgIuIiIiIiIiIiIg0Ap999hm9evUaF+TOlzzp378/Q4cOnWD7qqoq2rVrV8xgy0QBcBGRMsvqYqnhUHHUcEhERERERERkQn379uWII46Y7DZmxhxzzEGXLl3o0qULSy65JHPPPXf9DHAqUQBcRKTMsuCrGg4VQw2HRERERERERCbWqVMn7rjjDn777bdqm15mt549e9KzZ09eeeUVAG666Sbmm2++gkdfewqAi4iIiIiIiIiIiDQCc845J3POOedktxk2bBgDBgzgvffe48orr2Tw4MH1NLqpQwFwEREREREREREREQGgVatWtGrVqsGXPslUFT0AEREREREREREREZGpQQFwEREREREREREREalICoCLiIiIiIiIiIiISEVSAFxEREREREREREREKpIC4CIiIiIiIiIiIiJSkRQAFxEREREREREREZGKpAC4iIiIiIiIiIiIiFQkBcBFREREREREREREpCIpAC4iIiIiIiIiIiIiFUkBcBERERERERERERGpSAqAi4iIiIiIiIiIiEhFalr0AERERERERERERESk/o0cOZIBAwbQv3//cbfs37/88kvRwysLBcBFREREREREREREKtywYcO49tpr6d2797hg9+DBg6vddvrpp6d9+/asuOKKzDPPPPU80vJSAFxERERERETKrmvXrrz66qtFD6NR69GjR9FDEBGRacioUaP44Ycf6NWrF0OGDKl2GzNj9tlnp2PHjnTo0IHOnTvTsmXLeh5peSkALiIiIiIiImXXo0cPjjzyyKKH0Wh179696CGIiMg0Zvrpp+e6664DYMSIEROVPunfvz89e/bkxx9/5OOPPx73vGWXXZYllliiqGHXmQLgIiIiIiIiIiIiIhXO3fnoo4/o1avXBMHv7OcBAwYwatSoCZ7TtGlT2rVrV8yAy0QBcBEREREREZEKpDI0xVIJGhGZ1vz111+ccMIJjB07dpLbmBlzzDEHXbp0oUuXLiyxxBLMOeec9TjK8lMAXERERERERKQCqQxNcVSCRkSmRZ06deKuu+7i999/n6DsSWkplJ49e9KzZ09eeeUVAG688Ubmn3/+gkdfewqAi4iIiIiIiIiIiDQCs846K7POOutktxk9ejQDBw7k3Xff5dJLL51kw8yGoqroAYiIiIiIiIiIiIjItKFp06bMNNNMzDbbbEUPpSyUAS4iIiIiIiIiIiLSiLg7Q4YMqbYESvbvPn36FD3MslAAXERERERERERERKTCDR48mDPPPJPevXvTv39/Ro4cOdE2ZsaMM85I+/btmXXWWVlmmWVYcMEFCxht+SgALiIiIiIiImXXtWtXXn311aKH0aj16NGj6CGIiMg0xMxo06YNbdq0YcSIEYwaNQp3n2CbLDO8efPmtGjRgiFDhjBmzJiCRlweCoCLiIiIiIhI2fXo0YMjjzyy6GE0Wt27dy96CCIiMo1p06YN3bp1G/fvMWPGMGjQoGrLn/Tv359evXrx4osvsvHGG9O1a9fCxl1XCoCLiIiIiIiIiIiINCLuzujRo6mqqqJNmzY0adKEtm3b0rFjR4YNG8awYcPo0KED33//fdFDrTMFwEVEREREREREREQaGHfn448/5o8//mDYsGEMHz58XPA6u5Xel//32LFj//P/0axZM2acccZ6+G2mHgXARUREpOxU97VYqvkqIiKg43HRdDwWkaltwIABnHDCCXWu0d22bVs6dOgwwa19+/a0b9+eOeaYg9lnn71MIy6GAuAiIiJSdqr7WizVfRUREdDxuEg6FotIfejQoQP33Xcf/fv3nyjze0ozwrNbr1696NmzZ7X/nxtuuIEFFlignn+78lEAXERERERERERERKQB6tChA61atZriQPfk7h80aBD//PPPRP+P0aNHF/CblY8C4CIVJuvKq6WOxdNnUBwtNxUREREREZFKN2jQIHbfffdqg9Y1kZVAmXfeeceVQJlhhhlo3bo17du3Z8EFFyzTiIuhALiIiIiIiIiIiIhIA9O2bVv22msv+vbtO8XlT9x9otcZPHgwgwcPnqgESsuWLenQoQOLLroonTp1qq9fq+wUABepMFnmq2r9FSer96fPoBiqtygiIiIiIiKNQZMmTdhyyy2neHt3Z8SIEeMC4kOHDuXvv/+mT58+9OnTh99++43ffvuNPn36MHr0aIYPHz7uMQXARURERERERERERGSaNWzYMK644gp69+5N//796d+/PyNHjpxouyZNmjDTTDPRvn17OnfuTJcuXQoYbfkoAC4iIiIiIiJl17VrV/VEKZj6ooiISN6YMWP4448/+P333xk4cGC15VDMjJlnnnlcLfDOnTvTrFmzAkZbPgqAi4iIiIiISNn16NFDJeEKpLJwIiJSarrppuOKK64AIhg+aNCgcZng/fv3Z8CAARP8/N133/H666+z0korseSSSxY8+tpTAFxERERERERERESkEWnSpMm4LO9J+eSTTzj66KOrzRRvSKqKHoCIiIiIiIiIiIiIyNSgALiIiIiI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\n",
      "text/plain": [
       "<Figure size 1800x720 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "import missingno\n",
    "missingno.matrix(df)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Male      393\n",
       "Female     88\n",
       "Name: Gender, dtype: int64"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.Gender.value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Loan_ID</th>\n",
       "      <th>Gender</th>\n",
       "      <th>Married</th>\n",
       "      <th>Dependents</th>\n",
       "      <th>Education</th>\n",
       "      <th>Self_Employed</th>\n",
       "      <th>ApplicantIncome</th>\n",
       "      <th>CoapplicantIncome</th>\n",
       "      <th>LoanAmount</th>\n",
       "      <th>Loan_Amount_Term</th>\n",
       "      <th>Credit_History</th>\n",
       "      <th>Property_Area</th>\n",
       "      <th>Loan_Status</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>59</th>\n",
       "      <td>LP001585</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Yes</td>\n",
       "      <td>3+</td>\n",
       "      <td>Graduate</td>\n",
       "      <td>No</td>\n",
       "      <td>51763</td>\n",
       "      <td>0.0</td>\n",
       "      <td>700.0</td>\n",
       "      <td>300.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>Urban</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>125</th>\n",
       "      <td>LP002625</td>\n",
       "      <td>NaN</td>\n",
       "      <td>No</td>\n",
       "      <td>0</td>\n",
       "      <td>Graduate</td>\n",
       "      <td>No</td>\n",
       "      <td>3583</td>\n",
       "      <td>0.0</td>\n",
       "      <td>96.0</td>\n",
       "      <td>360.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>Urban</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>133</th>\n",
       "      <td>LP002872</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Yes</td>\n",
       "      <td>0</td>\n",
       "      <td>Graduate</td>\n",
       "      <td>No</td>\n",
       "      <td>3087</td>\n",
       "      <td>2210.0</td>\n",
       "      <td>136.0</td>\n",
       "      <td>360.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>Semiurban</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>243</th>\n",
       "      <td>LP002478</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Yes</td>\n",
       "      <td>0</td>\n",
       "      <td>Graduate</td>\n",
       "      <td>Yes</td>\n",
       "      <td>2083</td>\n",
       "      <td>4083.0</td>\n",
       "      <td>160.0</td>\n",
       "      <td>360.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Semiurban</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>360</th>\n",
       "      <td>LP001448</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Yes</td>\n",
       "      <td>3+</td>\n",
       "      <td>Graduate</td>\n",
       "      <td>No</td>\n",
       "      <td>23803</td>\n",
       "      <td>0.0</td>\n",
       "      <td>370.0</td>\n",
       "      <td>360.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>Rural</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>410</th>\n",
       "      <td>LP002501</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Yes</td>\n",
       "      <td>0</td>\n",
       "      <td>Graduate</td>\n",
       "      <td>No</td>\n",
       "      <td>16692</td>\n",
       "      <td>0.0</td>\n",
       "      <td>110.0</td>\n",
       "      <td>360.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>Semiurban</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>465</th>\n",
       "      <td>LP001644</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Yes</td>\n",
       "      <td>0</td>\n",
       "      <td>Graduate</td>\n",
       "      <td>Yes</td>\n",
       "      <td>674</td>\n",
       "      <td>5296.0</td>\n",
       "      <td>168.0</td>\n",
       "      <td>360.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>Rural</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>470</th>\n",
       "      <td>LP002530</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Yes</td>\n",
       "      <td>2</td>\n",
       "      <td>Graduate</td>\n",
       "      <td>No</td>\n",
       "      <td>2873</td>\n",
       "      <td>1872.0</td>\n",
       "      <td>132.0</td>\n",
       "      <td>360.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>Semiurban</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>475</th>\n",
       "      <td>LP002925</td>\n",
       "      <td>NaN</td>\n",
       "      <td>No</td>\n",
       "      <td>0</td>\n",
       "      <td>Graduate</td>\n",
       "      <td>No</td>\n",
       "      <td>4750</td>\n",
       "      <td>0.0</td>\n",
       "      <td>94.0</td>\n",
       "      <td>360.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>Semiurban</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>486</th>\n",
       "      <td>LP002103</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Yes</td>\n",
       "      <td>1</td>\n",
       "      <td>Graduate</td>\n",
       "      <td>Yes</td>\n",
       "      <td>9833</td>\n",
       "      <td>1833.0</td>\n",
       "      <td>182.0</td>\n",
       "      <td>180.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>Urban</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "      Loan_ID Gender Married Dependents Education Self_Employed  \\\n",
       "59   LP001585    NaN     Yes         3+  Graduate            No   \n",
       "125  LP002625    NaN      No          0  Graduate            No   \n",
       "133  LP002872    NaN     Yes          0  Graduate            No   \n",
       "243  LP002478    NaN     Yes          0  Graduate           Yes   \n",
       "360  LP001448    NaN     Yes         3+  Graduate            No   \n",
       "410  LP002501    NaN     Yes          0  Graduate            No   \n",
       "465  LP001644    NaN     Yes          0  Graduate           Yes   \n",
       "470  LP002530    NaN     Yes          2  Graduate            No   \n",
       "475  LP002925    NaN      No          0  Graduate            No   \n",
       "486  LP002103    NaN     Yes          1  Graduate           Yes   \n",
       "\n",
       "     ApplicantIncome  CoapplicantIncome  LoanAmount  Loan_Amount_Term  \\\n",
       "59             51763                0.0       700.0             300.0   \n",
       "125             3583                0.0        96.0             360.0   \n",
       "133             3087             2210.0       136.0             360.0   \n",
       "243             2083             4083.0       160.0             360.0   \n",
       "360            23803                0.0       370.0             360.0   \n",
       "410            16692                0.0       110.0             360.0   \n",
       "465              674             5296.0       168.0             360.0   \n",
       "470             2873             1872.0       132.0             360.0   \n",
       "475             4750                0.0        94.0             360.0   \n",
       "486             9833             1833.0       182.0             180.0   \n",
       "\n",
       "     Credit_History Property_Area  Loan_Status  \n",
       "59              1.0         Urban            1  \n",
       "125             1.0         Urban            0  \n",
       "133             0.0     Semiurban            0  \n",
       "243             NaN     Semiurban            1  \n",
       "360             1.0         Rural            1  \n",
       "410             1.0     Semiurban            1  \n",
       "465             1.0         Rural            1  \n",
       "470             0.0     Semiurban            0  \n",
       "475             1.0     Semiurban            1  \n",
       "486             1.0         Urban            1  "
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df[df['Gender'].isnull()]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [],
   "source": [
    "df.Gender.fillna(\"Male\", inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Loan_ID</th>\n",
       "      <th>Gender</th>\n",
       "      <th>Married</th>\n",
       "      <th>Dependents</th>\n",
       "      <th>Education</th>\n",
       "      <th>Self_Employed</th>\n",
       "      <th>ApplicantIncome</th>\n",
       "      <th>CoapplicantIncome</th>\n",
       "      <th>LoanAmount</th>\n",
       "      <th>Loan_Amount_Term</th>\n",
       "      <th>Credit_History</th>\n",
       "      <th>Property_Area</th>\n",
       "      <th>Loan_Status</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>41</th>\n",
       "      <td>LP001760</td>\n",
       "      <td>Male</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Graduate</td>\n",
       "      <td>No</td>\n",
       "      <td>4758</td>\n",
       "      <td>0.0</td>\n",
       "      <td>158.0</td>\n",
       "      <td>480.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>Semiurban</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "     Loan_ID Gender Married Dependents Education Self_Employed  \\\n",
       "41  LP001760   Male     NaN        NaN  Graduate            No   \n",
       "\n",
       "    ApplicantIncome  CoapplicantIncome  LoanAmount  Loan_Amount_Term  \\\n",
       "41             4758                0.0       158.0             480.0   \n",
       "\n",
       "    Credit_History Property_Area  Loan_Status  \n",
       "41             1.0     Semiurban            1  "
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df[df['Married'].isnull()]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [],
   "source": [
    "df.Married.fillna(\"No\", inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Loan_ID</th>\n",
       "      <th>Gender</th>\n",
       "      <th>Married</th>\n",
       "      <th>Dependents</th>\n",
       "      <th>Education</th>\n",
       "      <th>Self_Employed</th>\n",
       "      <th>ApplicantIncome</th>\n",
       "      <th>CoapplicantIncome</th>\n",
       "      <th>LoanAmount</th>\n",
       "      <th>Loan_Amount_Term</th>\n",
       "      <th>Credit_History</th>\n",
       "      <th>Property_Area</th>\n",
       "      <th>Loan_Status</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>11</th>\n",
       "      <td>LP002130</td>\n",
       "      <td>Male</td>\n",
       "      <td>Yes</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Not Graduate</td>\n",
       "      <td>No</td>\n",
       "      <td>3523</td>\n",
       "      <td>3230.0</td>\n",
       "      <td>152.0</td>\n",
       "      <td>360.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>Rural</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>41</th>\n",
       "      <td>LP001760</td>\n",
       "      <td>Male</td>\n",
       "      <td>No</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Graduate</td>\n",
       "      <td>No</td>\n",
       "      <td>4758</td>\n",
       "      <td>0.0</td>\n",
       "      <td>158.0</td>\n",
       "      <td>480.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>Semiurban</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>55</th>\n",
       "      <td>LP002943</td>\n",
       "      <td>Male</td>\n",
       "      <td>No</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Graduate</td>\n",
       "      <td>No</td>\n",
       "      <td>2987</td>\n",
       "      <td>0.0</td>\n",
       "      <td>88.0</td>\n",
       "      <td>360.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>Semiurban</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>127</th>\n",
       "      <td>LP001426</td>\n",
       "      <td>Male</td>\n",
       "      <td>Yes</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Graduate</td>\n",
       "      <td>No</td>\n",
       "      <td>5667</td>\n",
       "      <td>2667.0</td>\n",
       "      <td>180.0</td>\n",
       "      <td>360.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>Rural</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>281</th>\n",
       "      <td>LP002847</td>\n",
       "      <td>Male</td>\n",
       "      <td>Yes</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Graduate</td>\n",
       "      <td>No</td>\n",
       "      <td>5116</td>\n",
       "      <td>1451.0</td>\n",
       "      <td>165.0</td>\n",
       "      <td>360.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>Urban</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>348</th>\n",
       "      <td>LP001945</td>\n",
       "      <td>Female</td>\n",
       "      <td>No</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Graduate</td>\n",
       "      <td>No</td>\n",
       "      <td>5417</td>\n",
       "      <td>0.0</td>\n",
       "      <td>143.0</td>\n",
       "      <td>480.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>Urban</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>357</th>\n",
       "      <td>LP002144</td>\n",
       "      <td>Female</td>\n",
       "      <td>No</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Graduate</td>\n",
       "      <td>No</td>\n",
       "      <td>3813</td>\n",
       "      <td>0.0</td>\n",
       "      <td>116.0</td>\n",
       "      <td>180.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>Urban</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>407</th>\n",
       "      <td>LP002106</td>\n",
       "      <td>Male</td>\n",
       "      <td>Yes</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Graduate</td>\n",
       "      <td>Yes</td>\n",
       "      <td>5503</td>\n",
       "      <td>4490.0</td>\n",
       "      <td>70.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>1.0</td>\n",
       "      <td>Semiurban</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>453</th>\n",
       "      <td>LP001972</td>\n",
       "      <td>Male</td>\n",
       "      <td>Yes</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Not Graduate</td>\n",
       "      <td>No</td>\n",
       "      <td>2875</td>\n",
       "      <td>1750.0</td>\n",
       "      <td>105.0</td>\n",
       "      <td>360.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>Semiurban</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "      Loan_ID  Gender Married Dependents     Education Self_Employed  \\\n",
       "11   LP002130    Male     Yes        NaN  Not Graduate            No   \n",
       "41   LP001760    Male      No        NaN      Graduate            No   \n",
       "55   LP002943    Male      No        NaN      Graduate            No   \n",
       "127  LP001426    Male     Yes        NaN      Graduate            No   \n",
       "281  LP002847    Male     Yes        NaN      Graduate            No   \n",
       "348  LP001945  Female      No        NaN      Graduate            No   \n",
       "357  LP002144  Female      No        NaN      Graduate            No   \n",
       "407  LP002106    Male     Yes        NaN      Graduate           Yes   \n",
       "453  LP001972    Male     Yes        NaN  Not Graduate            No   \n",
       "\n",
       "     ApplicantIncome  CoapplicantIncome  LoanAmount  Loan_Amount_Term  \\\n",
       "11              3523             3230.0       152.0             360.0   \n",
       "41              4758                0.0       158.0             480.0   \n",
       "55              2987                0.0        88.0             360.0   \n",
       "127             5667             2667.0       180.0             360.0   \n",
       "281             5116             1451.0       165.0             360.0   \n",
       "348             5417                0.0       143.0             480.0   \n",
       "357             3813                0.0       116.0             180.0   \n",
       "407             5503             4490.0        70.0               NaN   \n",
       "453             2875             1750.0       105.0             360.0   \n",
       "\n",
       "     Credit_History Property_Area  Loan_Status  \n",
       "11              0.0         Rural            0  \n",
       "41              1.0     Semiurban            1  \n",
       "55              0.0     Semiurban            0  \n",
       "127             1.0         Rural            1  \n",
       "281             0.0         Urban            0  \n",
       "348             0.0         Urban            0  \n",
       "357             1.0         Urban            1  \n",
       "407             1.0     Semiurban            1  \n",
       "453             1.0     Semiurban            1  "
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df[df['Dependents'].isnull()]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[11, 41, 55, 127, 281, 348, 357, 407, 453]"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df[df['Dependents'].isnull()].index.tolist()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [],
   "source": [
    "df['Dependents']=df['Dependents'].str.replace(\"+\", \"\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [],
   "source": [
    "df.loc[11, \"Dependents\"]=1\n",
    "df.loc[127, \"Dependents\"]=1\n",
    "df.loc[281, \"Dependents\"]=1\n",
    "df.loc[407, \"Dependents\"]=1\n",
    "df.loc[453, \"Dependents\"]=1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [],
   "source": [
    "df.loc[55, \"Dependents\"]=0\n",
    "df.loc[348, \"Dependents\"]=0\n",
    "df.loc[357, \"Dependents\"]=0\n",
    "df.loc[41, \"Dependents\"]=0"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "False"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['Dependents'].isnull().any()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [],
   "source": [
    "df['Dependents']=df['Dependents'].astype(int)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "False"
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['Education'].isnull().any()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Graduate        388\n",
       "Not Graduate    103\n",
       "Name: Education, dtype: int64"
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['Education'].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Loan_ID</th>\n",
       "      <th>Gender</th>\n",
       "      <th>Married</th>\n",
       "      <th>Dependents</th>\n",
       "      <th>Education</th>\n",
       "      <th>Self_Employed</th>\n",
       "      <th>ApplicantIncome</th>\n",
       "      <th>CoapplicantIncome</th>\n",
       "      <th>LoanAmount</th>\n",
       "      <th>Loan_Amount_Term</th>\n",
       "      <th>Credit_History</th>\n",
       "      <th>Property_Area</th>\n",
       "      <th>Loan_Status</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>38</th>\n",
       "      <td>LP001052</td>\n",
       "      <td>Male</td>\n",
       "      <td>Yes</td>\n",
       "      <td>1</td>\n",
       "      <td>Graduate</td>\n",
       "      <td>NaN</td>\n",
       "      <td>3717</td>\n",
       "      <td>2925.0</td>\n",
       "      <td>151.0</td>\n",
       "      <td>360.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Semiurban</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>63</th>\n",
       "      <td>LP002435</td>\n",
       "      <td>Male</td>\n",
       "      <td>Yes</td>\n",
       "      <td>0</td>\n",
       "      <td>Graduate</td>\n",
       "      <td>NaN</td>\n",
       "      <td>3539</td>\n",
       "      <td>1376.0</td>\n",
       "      <td>55.0</td>\n",
       "      <td>360.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>Rural</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>79</th>\n",
       "      <td>LP002319</td>\n",
       "      <td>Male</td>\n",
       "      <td>Yes</td>\n",
       "      <td>0</td>\n",
       "      <td>Graduate</td>\n",
       "      <td>NaN</td>\n",
       "      <td>6256</td>\n",
       "      <td>0.0</td>\n",
       "      <td>160.0</td>\n",
       "      <td>360.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Urban</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>119</th>\n",
       "      <td>LP001091</td>\n",
       "      <td>Male</td>\n",
       "      <td>Yes</td>\n",
       "      <td>1</td>\n",
       "      <td>Graduate</td>\n",
       "      <td>NaN</td>\n",
       "      <td>4166</td>\n",
       "      <td>3369.0</td>\n",
       "      <td>201.0</td>\n",
       "      <td>360.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Urban</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>132</th>\n",
       "      <td>LP002732</td>\n",
       "      <td>Male</td>\n",
       "      <td>No</td>\n",
       "      <td>0</td>\n",
       "      <td>Not Graduate</td>\n",
       "      <td>NaN</td>\n",
       "      <td>2550</td>\n",
       "      <td>2042.0</td>\n",
       "      <td>126.0</td>\n",
       "      <td>360.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>Rural</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>164</th>\n",
       "      <td>LP002209</td>\n",
       "      <td>Female</td>\n",
       "      <td>No</td>\n",
       "      <td>0</td>\n",
       "      <td>Graduate</td>\n",
       "      <td>NaN</td>\n",
       "      <td>2764</td>\n",
       "      <td>1459.0</td>\n",
       "      <td>110.0</td>\n",
       "      <td>360.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>Urban</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>191</th>\n",
       "      <td>LP001883</td>\n",
       "      <td>Female</td>\n",
       "      <td>No</td>\n",
       "      <td>0</td>\n",
       "      <td>Graduate</td>\n",
       "      <td>NaN</td>\n",
       "      <td>3418</td>\n",
       "      <td>0.0</td>\n",
       "      <td>135.0</td>\n",
       "      <td>360.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>Rural</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>200</th>\n",
       "      <td>LP002489</td>\n",
       "      <td>Female</td>\n",
       "      <td>No</td>\n",
       "      <td>1</td>\n",
       "      <td>Not Graduate</td>\n",
       "      <td>NaN</td>\n",
       "      <td>5191</td>\n",
       "      <td>0.0</td>\n",
       "      <td>132.0</td>\n",
       "      <td>360.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>Semiurban</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>201</th>\n",
       "      <td>LP002950</td>\n",
       "      <td>Male</td>\n",
       "      <td>Yes</td>\n",
       "      <td>0</td>\n",
       "      <td>Not Graduate</td>\n",
       "      <td>NaN</td>\n",
       "      <td>2894</td>\n",
       "      <td>2792.0</td>\n",
       "      <td>155.0</td>\n",
       "      <td>360.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>Rural</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>205</th>\n",
       "      <td>LP001768</td>\n",
       "      <td>Male</td>\n",
       "      <td>Yes</td>\n",
       "      <td>0</td>\n",
       "      <td>Graduate</td>\n",
       "      <td>NaN</td>\n",
       "      <td>3716</td>\n",
       "      <td>0.0</td>\n",
       "      <td>42.0</td>\n",
       "      <td>180.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>Rural</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>226</th>\n",
       "      <td>LP001398</td>\n",
       "      <td>Male</td>\n",
       "      <td>No</td>\n",
       "      <td>0</td>\n",
       "      <td>Graduate</td>\n",
       "      <td>NaN</td>\n",
       "      <td>5050</td>\n",
       "      <td>0.0</td>\n",
       "      <td>118.0</td>\n",
       "      <td>360.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>Semiurban</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>229</th>\n",
       "      <td>LP002753</td>\n",
       "      <td>Female</td>\n",
       "      <td>No</td>\n",
       "      <td>1</td>\n",
       "      <td>Graduate</td>\n",
       "      <td>NaN</td>\n",
       "      <td>3652</td>\n",
       "      <td>0.0</td>\n",
       "      <td>95.0</td>\n",
       "      <td>360.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>Semiurban</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>257</th>\n",
       "      <td>LP001387</td>\n",
       "      <td>Female</td>\n",
       "      <td>Yes</td>\n",
       "      <td>0</td>\n",
       "      <td>Graduate</td>\n",
       "      <td>NaN</td>\n",
       "      <td>2929</td>\n",
       "      <td>2333.0</td>\n",
       "      <td>139.0</td>\n",
       "      <td>360.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>Semiurban</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>270</th>\n",
       "      <td>LP001041</td>\n",
       "      <td>Male</td>\n",
       "      <td>Yes</td>\n",
       "      <td>0</td>\n",
       "      <td>Graduate</td>\n",
       "      <td>NaN</td>\n",
       "      <td>2600</td>\n",
       "      <td>3500.0</td>\n",
       "      <td>115.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>1.0</td>\n",
       "      <td>Urban</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>285</th>\n",
       "      <td>LP001732</td>\n",
       "      <td>Male</td>\n",
       "      <td>Yes</td>\n",
       "      <td>2</td>\n",
       "      <td>Graduate</td>\n",
       "      <td>NaN</td>\n",
       "      <td>5000</td>\n",
       "      <td>0.0</td>\n",
       "      <td>72.0</td>\n",
       "      <td>360.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>Semiurban</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>287</th>\n",
       "      <td>LP002949</td>\n",
       "      <td>Female</td>\n",
       "      <td>No</td>\n",
       "      <td>3</td>\n",
       "      <td>Graduate</td>\n",
       "      <td>NaN</td>\n",
       "      <td>416</td>\n",
       "      <td>41667.0</td>\n",
       "      <td>350.0</td>\n",
       "      <td>180.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Urban</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>331</th>\n",
       "      <td>LP002110</td>\n",
       "      <td>Male</td>\n",
       "      <td>Yes</td>\n",
       "      <td>1</td>\n",
       "      <td>Graduate</td>\n",
       "      <td>NaN</td>\n",
       "      <td>5250</td>\n",
       "      <td>688.0</td>\n",
       "      <td>160.0</td>\n",
       "      <td>360.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>Rural</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>376</th>\n",
       "      <td>LP002101</td>\n",
       "      <td>Male</td>\n",
       "      <td>Yes</td>\n",
       "      <td>0</td>\n",
       "      <td>Graduate</td>\n",
       "      <td>NaN</td>\n",
       "      <td>63337</td>\n",
       "      <td>0.0</td>\n",
       "      <td>490.0</td>\n",
       "      <td>180.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>Urban</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>413</th>\n",
       "      <td>LP001786</td>\n",
       "      <td>Male</td>\n",
       "      <td>Yes</td>\n",
       "      <td>0</td>\n",
       "      <td>Graduate</td>\n",
       "      <td>NaN</td>\n",
       "      <td>5746</td>\n",
       "      <td>0.0</td>\n",
       "      <td>255.0</td>\n",
       "      <td>360.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Urban</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>421</th>\n",
       "      <td>LP002226</td>\n",
       "      <td>Male</td>\n",
       "      <td>Yes</td>\n",
       "      <td>0</td>\n",
       "      <td>Graduate</td>\n",
       "      <td>NaN</td>\n",
       "      <td>3333</td>\n",
       "      <td>2500.0</td>\n",
       "      <td>128.0</td>\n",
       "      <td>360.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>Semiurban</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>422</th>\n",
       "      <td>LP001326</td>\n",
       "      <td>Male</td>\n",
       "      <td>No</td>\n",
       "      <td>0</td>\n",
       "      <td>Graduate</td>\n",
       "      <td>NaN</td>\n",
       "      <td>6782</td>\n",
       "      <td>0.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>360.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Urban</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>423</th>\n",
       "      <td>LP001087</td>\n",
       "      <td>Female</td>\n",
       "      <td>No</td>\n",
       "      <td>2</td>\n",
       "      <td>Graduate</td>\n",
       "      <td>NaN</td>\n",
       "      <td>3750</td>\n",
       "      <td>2083.0</td>\n",
       "      <td>120.0</td>\n",
       "      <td>360.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>Semiurban</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>428</th>\n",
       "      <td>LP001027</td>\n",
       "      <td>Male</td>\n",
       "      <td>Yes</td>\n",
       "      <td>2</td>\n",
       "      <td>Graduate</td>\n",
       "      <td>NaN</td>\n",
       "      <td>2500</td>\n",
       "      <td>1840.0</td>\n",
       "      <td>109.0</td>\n",
       "      <td>360.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>Urban</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>432</th>\n",
       "      <td>LP001581</td>\n",
       "      <td>Male</td>\n",
       "      <td>Yes</td>\n",
       "      <td>0</td>\n",
       "      <td>Not Graduate</td>\n",
       "      <td>NaN</td>\n",
       "      <td>1820</td>\n",
       "      <td>1769.0</td>\n",
       "      <td>95.0</td>\n",
       "      <td>360.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>Rural</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>433</th>\n",
       "      <td>LP001949</td>\n",
       "      <td>Male</td>\n",
       "      <td>Yes</td>\n",
       "      <td>3</td>\n",
       "      <td>Graduate</td>\n",
       "      <td>NaN</td>\n",
       "      <td>4416</td>\n",
       "      <td>1250.0</td>\n",
       "      <td>110.0</td>\n",
       "      <td>360.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>Urban</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>436</th>\n",
       "      <td>LP001370</td>\n",
       "      <td>Male</td>\n",
       "      <td>No</td>\n",
       "      <td>0</td>\n",
       "      <td>Not Graduate</td>\n",
       "      <td>NaN</td>\n",
       "      <td>7333</td>\n",
       "      <td>0.0</td>\n",
       "      <td>120.0</td>\n",
       "      <td>360.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>Rural</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>437</th>\n",
       "      <td>LP002237</td>\n",
       "      <td>Male</td>\n",
       "      <td>No</td>\n",
       "      <td>1</td>\n",
       "      <td>Graduate</td>\n",
       "      <td>NaN</td>\n",
       "      <td>3667</td>\n",
       "      <td>0.0</td>\n",
       "      <td>113.0</td>\n",
       "      <td>180.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>Urban</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>469</th>\n",
       "      <td>LP001546</td>\n",
       "      <td>Male</td>\n",
       "      <td>No</td>\n",
       "      <td>0</td>\n",
       "      <td>Graduate</td>\n",
       "      <td>NaN</td>\n",
       "      <td>2980</td>\n",
       "      <td>2083.0</td>\n",
       "      <td>120.0</td>\n",
       "      <td>360.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>Rural</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>473</th>\n",
       "      <td>LP002888</td>\n",
       "      <td>Male</td>\n",
       "      <td>No</td>\n",
       "      <td>0</td>\n",
       "      <td>Graduate</td>\n",
       "      <td>NaN</td>\n",
       "      <td>3182</td>\n",
       "      <td>2917.0</td>\n",
       "      <td>161.0</td>\n",
       "      <td>360.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>Urban</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "      Loan_ID  Gender Married  Dependents     Education Self_Employed  \\\n",
       "38   LP001052    Male     Yes           1      Graduate           NaN   \n",
       "63   LP002435    Male     Yes           0      Graduate           NaN   \n",
       "79   LP002319    Male     Yes           0      Graduate           NaN   \n",
       "119  LP001091    Male     Yes           1      Graduate           NaN   \n",
       "132  LP002732    Male      No           0  Not Graduate           NaN   \n",
       "164  LP002209  Female      No           0      Graduate           NaN   \n",
       "191  LP001883  Female      No           0      Graduate           NaN   \n",
       "200  LP002489  Female      No           1  Not Graduate           NaN   \n",
       "201  LP002950    Male     Yes           0  Not Graduate           NaN   \n",
       "205  LP001768    Male     Yes           0      Graduate           NaN   \n",
       "226  LP001398    Male      No           0      Graduate           NaN   \n",
       "229  LP002753  Female      No           1      Graduate           NaN   \n",
       "257  LP001387  Female     Yes           0      Graduate           NaN   \n",
       "270  LP001041    Male     Yes           0      Graduate           NaN   \n",
       "285  LP001732    Male     Yes           2      Graduate           NaN   \n",
       "287  LP002949  Female      No           3      Graduate           NaN   \n",
       "331  LP002110    Male     Yes           1      Graduate           NaN   \n",
       "376  LP002101    Male     Yes           0      Graduate           NaN   \n",
       "413  LP001786    Male     Yes           0      Graduate           NaN   \n",
       "421  LP002226    Male     Yes           0      Graduate           NaN   \n",
       "422  LP001326    Male      No           0      Graduate           NaN   \n",
       "423  LP001087  Female      No           2      Graduate           NaN   \n",
       "428  LP001027    Male     Yes           2      Graduate           NaN   \n",
       "432  LP001581    Male     Yes           0  Not Graduate           NaN   \n",
       "433  LP001949    Male     Yes           3      Graduate           NaN   \n",
       "436  LP001370    Male      No           0  Not Graduate           NaN   \n",
       "437  LP002237    Male      No           1      Graduate           NaN   \n",
       "469  LP001546    Male      No           0      Graduate           NaN   \n",
       "473  LP002888    Male      No           0      Graduate           NaN   \n",
       "\n",
       "     ApplicantIncome  CoapplicantIncome  LoanAmount  Loan_Amount_Term  \\\n",
       "38              3717             2925.0       151.0             360.0   \n",
       "63              3539             1376.0        55.0             360.0   \n",
       "79              6256                0.0       160.0             360.0   \n",
       "119             4166             3369.0       201.0             360.0   \n",
       "132             2550             2042.0       126.0             360.0   \n",
       "164             2764             1459.0       110.0             360.0   \n",
       "191             3418                0.0       135.0             360.0   \n",
       "200             5191                0.0       132.0             360.0   \n",
       "201             2894             2792.0       155.0             360.0   \n",
       "205             3716                0.0        42.0             180.0   \n",
       "226             5050                0.0       118.0             360.0   \n",
       "229             3652                0.0        95.0             360.0   \n",
       "257             2929             2333.0       139.0             360.0   \n",
       "270             2600             3500.0       115.0               NaN   \n",
       "285             5000                0.0        72.0             360.0   \n",
       "287              416            41667.0       350.0             180.0   \n",
       "331             5250              688.0       160.0             360.0   \n",
       "376            63337                0.0       490.0             180.0   \n",
       "413             5746                0.0       255.0             360.0   \n",
       "421             3333             2500.0       128.0             360.0   \n",
       "422             6782                0.0         NaN             360.0   \n",
       "423             3750             2083.0       120.0             360.0   \n",
       "428             2500             1840.0       109.0             360.0   \n",
       "432             1820             1769.0        95.0             360.0   \n",
       "433             4416             1250.0       110.0             360.0   \n",
       "436             7333                0.0       120.0             360.0   \n",
       "437             3667                0.0       113.0             180.0   \n",
       "469             2980             2083.0       120.0             360.0   \n",
       "473             3182             2917.0       161.0             360.0   \n",
       "\n",
       "     Credit_History Property_Area  Loan_Status  \n",
       "38              NaN     Semiurban            0  \n",
       "63              1.0         Rural            0  \n",
       "79              NaN         Urban            1  \n",
       "119             NaN         Urban            0  \n",
       "132             1.0         Rural            1  \n",
       "164             1.0         Urban            1  \n",
       "191             1.0         Rural            0  \n",
       "200             1.0     Semiurban            1  \n",
       "201             1.0         Rural            1  \n",
       "205             1.0         Rural            1  \n",
       "226             1.0     Semiurban            1  \n",
       "229             1.0     Semiurban            1  \n",
       "257             1.0     Semiurban            1  \n",
       "270             1.0         Urban            1  \n",
       "285             0.0     Semiurban            0  \n",
       "287             NaN         Urban            0  \n",
       "331             1.0         Rural            1  \n",
       "376             1.0         Urban            1  \n",
       "413             NaN         Urban            0  \n",
       "421             1.0     Semiurban            1  \n",
       "422             NaN         Urban            0  \n",
       "423             1.0     Semiurban            1  \n",
       "428             1.0         Urban            1  \n",
       "432             1.0         Rural            1  \n",
       "433             1.0         Urban            1  \n",
       "436             1.0         Rural            0  \n",
       "437             1.0         Urban            1  \n",
       "469             1.0         Rural            1  \n",
       "473             1.0         Urban            1  "
      ]
     },
     "execution_count": 24,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df[df['Self_Employed'].isnull()]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "No     398\n",
       "Yes     64\n",
       "NaN     29\n",
       "Name: Self_Employed, dtype: int64"
      ]
     },
     "execution_count": 25,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['Self_Employed'].value_counts(dropna=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [],
   "source": [
    "df['Self_Employed'].fillna(\"No\", inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0"
      ]
     },
     "execution_count": 27,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.ApplicantIncome.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0"
      ]
     },
     "execution_count": 28,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.CoapplicantIncome.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "16"
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.LoanAmount.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [],
   "source": [
    "df.LoanAmount=df.LoanAmount.fillna(df.LoanAmount.median())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0"
      ]
     },
     "execution_count": 31,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.LoanAmount.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "dtype('float64')"
      ]
     },
     "execution_count": 32,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.LoanAmount.dtypes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "360.0    404\n",
       "180.0     35\n",
       "480.0     13\n",
       "300.0     12\n",
       "84.0       4\n",
       "120.0      3\n",
       "240.0      3\n",
       "36.0       2\n",
       "60.0       1\n",
       "12.0       1\n",
       "Name: Loan_Amount_Term, dtype: int64"
      ]
     },
     "execution_count": 33,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.Loan_Amount_Term.value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {},
   "outputs": [],
   "source": [
    "df.Loan_Amount_Term.fillna(df.Loan_Amount_Term.median(), inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "43"
      ]
     },
     "execution_count": 35,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.Credit_History.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "1.0    380\n",
       "0.0     68\n",
       "Name: Credit_History, dtype: int64"
      ]
     },
     "execution_count": 36,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.Credit_History.value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {},
   "outputs": [],
   "source": [
    "df.Credit_History.fillna(1, limit=18, inplace=True)\n",
    "df.Credit_History.fillna(0, limit=3, inplace=True)\n",
    "df.Credit_History.fillna(1, limit=18, inplace=True)\n",
    "df.Credit_History.fillna(0, limit=4, inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "dtype('float64')"
      ]
     },
     "execution_count": 38,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.Credit_History.dtype"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0"
      ]
     },
     "execution_count": 39,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.Property_Area.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "1    343\n",
       "0    148\n",
       "Name: Loan_Status, dtype: int64"
      ]
     },
     "execution_count": 40,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.Loan_Status.value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Semiurban    186\n",
       "Urban        155\n",
       "Rural        150\n",
       "Name: Property_Area, dtype: int64"
      ]
     },
     "execution_count": 41,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.Property_Area.value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "Int64Index: 491 entries, 0 to 490\n",
      "Data columns (total 13 columns):\n",
      " #   Column             Non-Null Count  Dtype  \n",
      "---  ------             --------------  -----  \n",
      " 0   Loan_ID            491 non-null    object \n",
      " 1   Gender             491 non-null    object \n",
      " 2   Married            491 non-null    object \n",
      " 3   Dependents         491 non-null    int32  \n",
      " 4   Education          491 non-null    object \n",
      " 5   Self_Employed      491 non-null    object \n",
      " 6   ApplicantIncome    491 non-null    int64  \n",
      " 7   CoapplicantIncome  491 non-null    float64\n",
      " 8   LoanAmount         491 non-null    float64\n",
      " 9   Loan_Amount_Term   491 non-null    float64\n",
      " 10  Credit_History     491 non-null    float64\n",
      " 11  Property_Area      491 non-null    object \n",
      " 12  Loan_Status        491 non-null    int64  \n",
      "dtypes: float64(4), int32(1), int64(2), object(6)\n",
      "memory usage: 71.8+ KB\n"
     ]
    }
   ],
   "source": [
    "df.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "metadata": {},
   "outputs": [],
   "source": [
    "df.drop(\"Loan_ID\", axis=1, inplace=True)\n",
    "#df.drop([\"Loan_ID\",  \"Loan_Amount_Term\", \"LoanAmount\", \"ApplicantIncome\", \"Dependents\", \"CoapplicantIncome\", \"Self_Employed\", \"Education\", \"Property_Area\"],axis=1, inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Dependents</th>\n",
       "      <th>ApplicantIncome</th>\n",
       "      <th>CoapplicantIncome</th>\n",
       "      <th>LoanAmount</th>\n",
       "      <th>Loan_Amount_Term</th>\n",
       "      <th>Credit_History</th>\n",
       "      <th>Loan_Status</th>\n",
       "      <th>Gender_Male</th>\n",
       "      <th>Married_Yes</th>\n",
       "      <th>Education_Not Graduate</th>\n",
       "      <th>Self_Employed_Yes</th>\n",
       "      <th>Property_Area_Semiurban</th>\n",
       "      <th>Property_Area_Urban</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "      <td>4547</td>\n",
       "      <td>0.0</td>\n",
       "      <td>115.0</td>\n",
       "      <td>360.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>3</td>\n",
       "      <td>5703</td>\n",
       "      <td>0.0</td>\n",
       "      <td>130.0</td>\n",
       "      <td>360.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>0</td>\n",
       "      <td>4333</td>\n",
       "      <td>2451.0</td>\n",
       "      <td>110.0</td>\n",
       "      <td>360.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>0</td>\n",
       "      <td>4695</td>\n",
       "      <td>0.0</td>\n",
       "      <td>96.0</td>\n",
       "      <td>360.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>2</td>\n",
       "      <td>6700</td>\n",
       "      <td>1750.0</td>\n",
       "      <td>230.0</td>\n",
       "      <td>300.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Dependents  ApplicantIncome  CoapplicantIncome  LoanAmount  \\\n",
       "0           0             4547                0.0       115.0   \n",
       "1           3             5703                0.0       130.0   \n",
       "2           0             4333             2451.0       110.0   \n",
       "3           0             4695                0.0        96.0   \n",
       "4           2             6700             1750.0       230.0   \n",
       "\n",
       "   Loan_Amount_Term  Credit_History  Loan_Status  Gender_Male  Married_Yes  \\\n",
       "0             360.0             1.0            1            0            0   \n",
       "1             360.0             1.0            1            1            1   \n",
       "2             360.0             1.0            0            0            1   \n",
       "3             360.0             1.0            1            1            1   \n",
       "4             300.0             1.0            1            1            1   \n",
       "\n",
       "   Education_Not Graduate  Self_Employed_Yes  Property_Area_Semiurban  \\\n",
       "0                       0                  0                        1   \n",
       "1                       1                  1                        0   \n",
       "2                       0                  0                        0   \n",
       "3                       1                  1                        0   \n",
       "4                       0                  0                        1   \n",
       "\n",
       "   Property_Area_Urban  \n",
       "0                    0  \n",
       "1                    0  \n",
       "2                    1  \n",
       "3                    1  \n",
       "4                    0  "
      ]
     },
     "execution_count": 44,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df=pd.get_dummies(df, drop_first=True)\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "metadata": {},
   "outputs": [],
   "source": [
    "X=df.drop(\"Loan_Status\", axis=1)\n",
    "y=df[\"Loan_Status\"]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.3, random_state=42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "metadata": {},
   "outputs": [],
   "source": [
    "#from sklearn.preprocessing import StandardScaler\n",
    "#sc=StandardScaler()\n",
    "#X_train_s=sc.fit_transform(X_train)\n",
    "#X_test_s=sc.transform(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Num Features: 6\n",
      "Selected Features: [ True  True  True  True  True  True False False False False False False]\n",
      "Feature Ranking: [1 1 1 1 1 1 4 3 6 7 2 5]\n"
     ]
    }
   ],
   "source": [
    "from sklearn.feature_selection import RFE #importing RFE class from sklearn library\n",
    "\n",
    "rfe = RFE(RandomForestClassifier() , step = 1) \n",
    "# estimator clf_lr is the baseline model (basic model) that we have created under \"Base line Model\" selection\n",
    "# step = 1: removes one feature at a time and then builds a model on the remaining features\n",
    "# It uses the model accuracy to identify which features (and combination of features) contribute the most to predicting the target variable.\n",
    "# we can even provide no. of features as an argument \n",
    "\n",
    "# Fit the function for ranking the features\n",
    "fit = rfe.fit(X_train, y_train)\n",
    "\n",
    "print(\"Num Features: %d\" % fit.n_features_)\n",
    "print(\"Selected Features: %s\" % fit.support_)\n",
    "print(\"Feature Ranking: %s\" % fit.ranking_)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "\n",
    "selected_rfe_features = pd.DataFrame({'Feature':list(X_train.columns),\n",
    "                                      'Ranking':rfe.ranking_})\n",
    "selected_rfe_features.sort_values(by='Ranking')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train_r=rfe.transform(X_train)\n",
    "X_test_r=rfe.transform(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.tree import DecisionTreeClassifier\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.svm import SVC\n",
    "from sklearn.ensemble import GradientBoostingClassifier \n",
    "from sklearn.model_selection import cross_val_score\n",
    "from xgboost import XGBClassifier"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "random_forest F1 score : 0.8471369943326167\n",
      "logistic_reg F1 score : 0.8618886734491257\n",
      "XGB F1 score : 0.8390180643339891\n",
      "GB F1 score : 0.8228854704573786\n",
      "Decision Tree F1 score : 0.7579943696417534\n"
     ]
    }
   ],
   "source": [
    "def model_check(models, X_train, y_train):\n",
    "    for name, model in models.items():\n",
    "        score = cross_val_score(model, X_train, y_train, cv=3, scoring='f1', n_jobs=-1)\n",
    "        print(f'{name} F1 score : {np.mean(score)}')\n",
    "\n",
    "models = {'random_forest':RandomForestClassifier(), \n",
    "          'logistic_reg':LogisticRegression(), \n",
    "          'XGB':XGBClassifier(), \n",
    "          'GB':GradientBoostingClassifier(),\n",
    "           \"Decision Tree\": DecisionTreeClassifier()}\n",
    "model_check(models, X_train_r, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 110,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.8938053097345134\n"
     ]
    }
   ],
   "source": [
    "logreg=LogisticRegression(C=5).fit(X_train_r,y_train)\n",
    "y_predl=logreg.predict(X_test_r)\n",
    "from sklearn.metrics import f1_score\n",
    "print(f1_score(y_test, y_predl))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 111,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.8938053097345134\n"
     ]
    }
   ],
   "source": [
    "rf=RandomForestClassifier(max_features=3,  min_samples_split=5, n_estimators=250, max_depth=2).fit(X_train_r, y_train)\n",
    "y_predr=rf.predict(X_test_r)\n",
    "print(f1_score(y_test, y_predr))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 63,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Fitting 5 folds for each of 108 candidates, totalling 540 fits\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[Parallel(n_jobs=-1)]: Using backend LokyBackend with 4 concurrent workers.\n",
      "[Parallel(n_jobs=-1)]: Done  33 tasks      | elapsed:    5.8s\n",
      "[Parallel(n_jobs=-1)]: Done 154 tasks      | elapsed:   26.8s\n",
      "[Parallel(n_jobs=-1)]: Done 357 tasks      | elapsed:   57.8s\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "BEST PARAM {'max_depth': 3, 'max_features': 2, 'min_samples_split': 2, 'n_estimators': 50}\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[Parallel(n_jobs=-1)]: Done 540 out of 540 | elapsed:  1.5min finished\n",
      "[Parallel(n_jobs=-1)]: Using backend ThreadingBackend with 4 concurrent workers.\n",
      "[Parallel(n_jobs=-1)]: Done  42 tasks      | elapsed:    0.0s\n",
      "[Parallel(n_jobs=-1)]: Done  50 out of  50 | elapsed:    0.0s finished\n"
     ]
    }
   ],
   "source": [
    "from sklearn.model_selection import RandomizedSearchCV, GridSearchCV\n",
    "\n",
    "rf = RandomForestClassifier(n_jobs=-1, verbose=1)\n",
    "\n",
    "rf_params = {\"n_estimators\":[50,100,300], \"max_depth\":[3,5,7], \"max_features\":[2,4,6,8], \"min_samples_split\":[2,4,6]}\n",
    "rf_cv_model = GridSearchCV(rf, rf_params, cv = 5, n_jobs = -1, verbose = 2).fit(X_train_r, y_train)\n",
    "print('BEST PARAM', rf_cv_model.best_params_)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 64,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.8938053097345134\n"
     ]
    }
   ],
   "source": [
    "rf2=RandomForestClassifier(max_features=2,  min_samples_split=2, n_estimators=50, max_depth=3).fit(X_train_r, y_train)\n",
    "y_predr=rf2.predict(X_test_r)\n",
    "print(f1_score(y_test, y_predr))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 74,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.8387096774193549\n"
     ]
    }
   ],
   "source": [
    "xgb_model=XGBClassifier().fit(X_train_r,y_train)\n",
    "y_predx = xgb_model.predict(X_test_r)\n",
    "print(f1_score(y_test, y_predx))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 75,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Fitting 3 folds for each of 81 candidates, totalling 243 fits\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[Parallel(n_jobs=-1)]: Using backend LokyBackend with 4 concurrent workers.\n",
      "[Parallel(n_jobs=-1)]: Done  58 tasks      | elapsed:    1.7s\n",
      "[Parallel(n_jobs=-1)]: Done 243 out of 243 | elapsed:    8.3s finished\n"
     ]
    }
   ],
   "source": [
    "xgb_params = {\"n_estimators\": [50, 100, 300],\n",
    "             \"subsample\":[0.5,0.8,1],\n",
    "             \"max_depth\":[3,5,7],\n",
    "             \"learning_rate\":[0.1,0.01,0.3]}\n",
    "xgb_model = XGBClassifier().fit(X_train_r, y_train)\n",
    "xgb_cv_model = GridSearchCV(xgb_model, xgb_params, cv = 3,\n",
    "                            n_jobs = -1, verbose = 2).fit(X_train_r, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 76,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 50, 'subsample': 0.5}"
      ]
     },
     "execution_count": 76,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "xgb_cv_model.best_params_"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 77,
   "metadata": {},
   "outputs": [],
   "source": [
    "xgb_tuned = XGBClassifier(learning_rate= 0.01,\n",
    "                                max_depth= 3,\n",
    "                                n_estimators= 50,\n",
    "                                subsample= 0.5).fit(X_train_r, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 78,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.8938053097345134\n"
     ]
    }
   ],
   "source": [
    "y_predxt=xgb_tuned.predict(X_test_r)\n",
    "print(f1_score(y_test, y_predxt))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 81,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.8160000000000001\n"
     ]
    }
   ],
   "source": [
    "from sklearn.svm import SVC\n",
    "sv=SVC(C=0.2).fit(X_train_r, y_train)\n",
    "y_predsv=sv.predict(X_test_r)\n",
    "print(f1_score(y_test, y_predsv))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 82,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "D:\\Anaconda belgeler\\lib\\site-packages\\tpot\\builtins\\__init__.py:36: UserWarning: Warning: optional dependency `torch` is not available. - skipping import of NN models.\n",
      "  warnings.warn(\"Warning: optional dependency `torch` is not available. - skipping import of NN models.\")\n",
      "Version 0.11.6.post2 of tpot is outdated. Version 0.11.6.post3 was released Monday December 14, 2020.\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(HTML(value='Optimization Progress'), FloatProgress(value=0.0, max=1350.0), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "Generation 1 - Current best internal CV score: 0.8753585157221689\n",
      "\n",
      "Generation 2 - Current best internal CV score: 0.8775461521548948\n",
      "\n",
      "Generation 3 - Current best internal CV score: 0.8775461521548948\n",
      "\n",
      "Generation 4 - Current best internal CV score: 0.8776920512564077\n",
      "\n",
      "Generation 5 - Current best internal CV score: 0.8776920512564077\n",
      "\n",
      "Generation 6 - Current best internal CV score: 0.8776920512564077\n",
      "\n",
      "Generation 7 - Current best internal CV score: 0.8776920512564077\n",
      "\n",
      "Generation 8 - Current best internal CV score: 0.8793384828678945\n",
      "\n",
      "Best pipeline: BernoulliNB(MLPClassifier(GradientBoostingClassifier(LinearSVC(input_matrix, C=15.0, dual=True, loss=hinge, penalty=l2, tol=0.1), learning_rate=0.1, max_depth=8, max_features=0.3, min_samples_leaf=18, min_samples_split=16, n_estimators=100, subsample=0.15000000000000002), alpha=0.1, learning_rate_init=0.1), alpha=0.1, fit_prior=True)\n",
      "f1 score is 89.77777777777777%\n"
     ]
    }
   ],
   "source": [
    "from tpot import TPOTClassifier\n",
    "tpot = TPOTClassifier(generations=8, population_size=150, scoring=\"f1\", verbosity=2)\n",
    "tpot.fit(X_train_r, y_train)\n",
    "print(\"f1 score is {}%\".format(tpot.score(X_test_r, y_test)*100))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 83,
   "metadata": {},
   "outputs": [],
   "source": [
    "test= pd.read_csv('https://raw.githubusercontent.com/dphi-official/Datasets/master/Loan_Data/loan_test.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 84,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 123 entries, 0 to 122\n",
      "Data columns (total 12 columns):\n",
      " #   Column             Non-Null Count  Dtype  \n",
      "---  ------             --------------  -----  \n",
      " 0   Loan_ID            123 non-null    object \n",
      " 1   Gender             120 non-null    object \n",
      " 2   Married            121 non-null    object \n",
      " 3   Dependents         117 non-null    object \n",
      " 4   Education          123 non-null    object \n",
      " 5   Self_Employed      120 non-null    object \n",
      " 6   ApplicantIncome    123 non-null    int64  \n",
      " 7   CoapplicantIncome  123 non-null    float64\n",
      " 8   LoanAmount         117 non-null    float64\n",
      " 9   Loan_Amount_Term   122 non-null    float64\n",
      " 10  Credit_History     116 non-null    float64\n",
      " 11  Property_Area      123 non-null    object \n",
      "dtypes: float64(4), int64(1), object(7)\n",
      "memory usage: 11.7+ KB\n"
     ]
    }
   ],
   "source": [
    "test.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 85,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Loan_ID</th>\n",
       "      <th>Gender</th>\n",
       "      <th>Married</th>\n",
       "      <th>Dependents</th>\n",
       "      <th>Education</th>\n",
       "      <th>Self_Employed</th>\n",
       "      <th>ApplicantIncome</th>\n",
       "      <th>CoapplicantIncome</th>\n",
       "      <th>LoanAmount</th>\n",
       "      <th>Loan_Amount_Term</th>\n",
       "      <th>Credit_History</th>\n",
       "      <th>Property_Area</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>LP001116</td>\n",
       "      <td>Male</td>\n",
       "      <td>No</td>\n",
       "      <td>0</td>\n",
       "      <td>Not Graduate</td>\n",
       "      <td>No</td>\n",
       "      <td>3748</td>\n",
       "      <td>1668.0</td>\n",
       "      <td>110.0</td>\n",
       "      <td>360.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>Semiurban</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>LP001488</td>\n",
       "      <td>Male</td>\n",
       "      <td>Yes</td>\n",
       "      <td>3+</td>\n",
       "      <td>Graduate</td>\n",
       "      <td>No</td>\n",
       "      <td>4000</td>\n",
       "      <td>7750.0</td>\n",
       "      <td>290.0</td>\n",
       "      <td>360.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>Semiurban</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>LP002138</td>\n",
       "      <td>Male</td>\n",
       "      <td>Yes</td>\n",
       "      <td>0</td>\n",
       "      <td>Graduate</td>\n",
       "      <td>No</td>\n",
       "      <td>2625</td>\n",
       "      <td>6250.0</td>\n",
       "      <td>187.0</td>\n",
       "      <td>360.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>Rural</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>LP002284</td>\n",
       "      <td>Male</td>\n",
       "      <td>No</td>\n",
       "      <td>0</td>\n",
       "      <td>Not Graduate</td>\n",
       "      <td>No</td>\n",
       "      <td>3902</td>\n",
       "      <td>1666.0</td>\n",
       "      <td>109.0</td>\n",
       "      <td>360.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>Rural</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>LP002328</td>\n",
       "      <td>Male</td>\n",
       "      <td>Yes</td>\n",
       "      <td>0</td>\n",
       "      <td>Not Graduate</td>\n",
       "      <td>No</td>\n",
       "      <td>6096</td>\n",
       "      <td>0.0</td>\n",
       "      <td>218.0</td>\n",
       "      <td>360.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>Rural</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "    Loan_ID Gender Married Dependents     Education Self_Employed  \\\n",
       "0  LP001116   Male      No          0  Not Graduate            No   \n",
       "1  LP001488   Male     Yes         3+      Graduate            No   \n",
       "2  LP002138   Male     Yes          0      Graduate            No   \n",
       "3  LP002284   Male      No          0  Not Graduate            No   \n",
       "4  LP002328   Male     Yes          0  Not Graduate            No   \n",
       "\n",
       "   ApplicantIncome  CoapplicantIncome  LoanAmount  Loan_Amount_Term  \\\n",
       "0             3748             1668.0       110.0             360.0   \n",
       "1             4000             7750.0       290.0             360.0   \n",
       "2             2625             6250.0       187.0             360.0   \n",
       "3             3902             1666.0       109.0             360.0   \n",
       "4             6096                0.0       218.0             360.0   \n",
       "\n",
       "   Credit_History Property_Area  \n",
       "0             1.0     Semiurban  \n",
       "1             1.0     Semiurban  \n",
       "2             1.0         Rural  \n",
       "3             1.0         Rural  \n",
       "4             0.0         Rural  "
      ]
     },
     "execution_count": 85,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "test.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 86,
   "metadata": {},
   "outputs": [],
   "source": [
    "test.Gender.fillna(\"Male\", inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 87,
   "metadata": {},
   "outputs": [],
   "source": [
    "test.drop(\"Loan_ID\", axis=1, inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 88,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Gender</th>\n",
       "      <th>Married</th>\n",
       "      <th>Dependents</th>\n",
       "      <th>Education</th>\n",
       "      <th>Self_Employed</th>\n",
       "      <th>ApplicantIncome</th>\n",
       "      <th>CoapplicantIncome</th>\n",
       "      <th>LoanAmount</th>\n",
       "      <th>Loan_Amount_Term</th>\n",
       "      <th>Credit_History</th>\n",
       "      <th>Property_Area</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>29</th>\n",
       "      <td>Female</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Graduate</td>\n",
       "      <td>No</td>\n",
       "      <td>10047</td>\n",
       "      <td>0.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>240.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>Semiurban</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>48</th>\n",
       "      <td>Male</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Graduate</td>\n",
       "      <td>No</td>\n",
       "      <td>3816</td>\n",
       "      <td>754.0</td>\n",
       "      <td>160.0</td>\n",
       "      <td>360.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>Urban</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "    Gender Married Dependents Education Self_Employed  ApplicantIncome  \\\n",
       "29  Female     NaN        NaN  Graduate            No            10047   \n",
       "48    Male     NaN        NaN  Graduate            No             3816   \n",
       "\n",
       "    CoapplicantIncome  LoanAmount  Loan_Amount_Term  Credit_History  \\\n",
       "29                0.0         NaN             240.0             1.0   \n",
       "48              754.0       160.0             360.0             1.0   \n",
       "\n",
       "   Property_Area  \n",
       "29     Semiurban  \n",
       "48         Urban  "
      ]
     },
     "execution_count": 88,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "test[test.Married.isnull()]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 89,
   "metadata": {},
   "outputs": [],
   "source": [
    "test.loc[29, \"Married\"]=\"No\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 90,
   "metadata": {},
   "outputs": [],
   "source": [
    "test.loc[48, \"Married\"]=\"Yes\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 91,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Gender</th>\n",
       "      <th>Married</th>\n",
       "      <th>Dependents</th>\n",
       "      <th>Education</th>\n",
       "      <th>Self_Employed</th>\n",
       "      <th>ApplicantIncome</th>\n",
       "      <th>CoapplicantIncome</th>\n",
       "      <th>LoanAmount</th>\n",
       "      <th>Loan_Amount_Term</th>\n",
       "      <th>Credit_History</th>\n",
       "      <th>Property_Area</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>26</th>\n",
       "      <td>Male</td>\n",
       "      <td>Yes</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Graduate</td>\n",
       "      <td>No</td>\n",
       "      <td>13650</td>\n",
       "      <td>0.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>360.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>Urban</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>29</th>\n",
       "      <td>Female</td>\n",
       "      <td>No</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Graduate</td>\n",
       "      <td>No</td>\n",
       "      <td>10047</td>\n",
       "      <td>0.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>240.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>Semiurban</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>39</th>\n",
       "      <td>Male</td>\n",
       "      <td>Yes</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Not Graduate</td>\n",
       "      <td>Yes</td>\n",
       "      <td>4735</td>\n",
       "      <td>0.0</td>\n",
       "      <td>138.0</td>\n",
       "      <td>360.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>Urban</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>48</th>\n",
       "      <td>Male</td>\n",
       "      <td>Yes</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Graduate</td>\n",
       "      <td>No</td>\n",
       "      <td>3816</td>\n",
       "      <td>754.0</td>\n",
       "      <td>160.0</td>\n",
       "      <td>360.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>Urban</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>73</th>\n",
       "      <td>Male</td>\n",
       "      <td>Yes</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Not Graduate</td>\n",
       "      <td>No</td>\n",
       "      <td>3074</td>\n",
       "      <td>1800.0</td>\n",
       "      <td>123.0</td>\n",
       "      <td>360.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>Semiurban</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>106</th>\n",
       "      <td>Male</td>\n",
       "      <td>No</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Graduate</td>\n",
       "      <td>No</td>\n",
       "      <td>2833</td>\n",
       "      <td>0.0</td>\n",
       "      <td>71.0</td>\n",
       "      <td>360.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>Urban</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "     Gender Married Dependents     Education Self_Employed  ApplicantIncome  \\\n",
       "26     Male     Yes        NaN      Graduate            No            13650   \n",
       "29   Female      No        NaN      Graduate            No            10047   \n",
       "39     Male     Yes        NaN  Not Graduate           Yes             4735   \n",
       "48     Male     Yes        NaN      Graduate            No             3816   \n",
       "73     Male     Yes        NaN  Not Graduate            No             3074   \n",
       "106    Male      No        NaN      Graduate            No             2833   \n",
       "\n",
       "     CoapplicantIncome  LoanAmount  Loan_Amount_Term  Credit_History  \\\n",
       "26                 0.0         NaN             360.0             1.0   \n",
       "29                 0.0         NaN             240.0             1.0   \n",
       "39                 0.0       138.0             360.0             1.0   \n",
       "48               754.0       160.0             360.0             1.0   \n",
       "73              1800.0       123.0             360.0             0.0   \n",
       "106                0.0        71.0             360.0             1.0   \n",
       "\n",
       "    Property_Area  \n",
       "26          Urban  \n",
       "29      Semiurban  \n",
       "39          Urban  \n",
       "48          Urban  \n",
       "73      Semiurban  \n",
       "106         Urban  "
      ]
     },
     "execution_count": 91,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "test[test.Dependents.isnull()]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 92,
   "metadata": {},
   "outputs": [],
   "source": [
    "test.Dependents=test.Dependents.str.replace(\"+\", \"\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 93,
   "metadata": {},
   "outputs": [],
   "source": [
    "test.loc[26, \"Dependents\"]=1\n",
    "test.loc[29, \"Dependents\"]=0\n",
    "test.loc[39, \"Dependents\"]=1\n",
    "test.loc[48, \"Dependents\"]=1\n",
    "test.loc[73, \"Dependents\"]=1\n",
    "test.loc[103, \"Dependents\"]=1\n",
    "test.loc[106, \"Dependents\"]=0"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 94,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0"
      ]
     },
     "execution_count": 94,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "test.Dependents.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 95,
   "metadata": {},
   "outputs": [],
   "source": [
    "test.Self_Employed.fillna(\"No\", inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 96,
   "metadata": {},
   "outputs": [],
   "source": [
    "test.LoanAmount.fillna(test.LoanAmount.median(), inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 97,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "\n",
    "test.Loan_Amount_Term.fillna(test.Loan_Amount_Term.median(), inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 98,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "1.0    95\n",
       "0.0    21\n",
       "NaN     7\n",
       "Name: Credit_History, dtype: int64"
      ]
     },
     "execution_count": 98,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "test.Credit_History.value_counts(dropna=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 99,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "test.Credit_History.fillna(1, limit=5, inplace=True)\n",
    "test.Credit_History.fillna(0, limit=2, inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 100,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 123 entries, 0 to 122\n",
      "Data columns (total 11 columns):\n",
      " #   Column             Non-Null Count  Dtype  \n",
      "---  ------             --------------  -----  \n",
      " 0   Gender             123 non-null    object \n",
      " 1   Married            123 non-null    object \n",
      " 2   Dependents         123 non-null    object \n",
      " 3   Education          123 non-null    object \n",
      " 4   Self_Employed      123 non-null    object \n",
      " 5   ApplicantIncome    123 non-null    int64  \n",
      " 6   CoapplicantIncome  123 non-null    float64\n",
      " 7   LoanAmount         123 non-null    float64\n",
      " 8   Loan_Amount_Term   123 non-null    float64\n",
      " 9   Credit_History     123 non-null    float64\n",
      " 10  Property_Area      123 non-null    object \n",
      "dtypes: float64(4), int64(1), object(6)\n",
      "memory usage: 10.7+ KB\n"
     ]
    }
   ],
   "source": [
    "test.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 307,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 101,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "123"
      ]
     },
     "execution_count": 101,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "test.Dependents.count()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 102,
   "metadata": {},
   "outputs": [],
   "source": [
    "test.Dependents=test.Dependents.astype(int)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 98,
   "metadata": {},
   "outputs": [],
   "source": [
    "#test.drop([\"Loan_Amount_Term\", \"LoanAmount\", \"ApplicantIncome\", \"Dependents\", \"CoapplicantIncome\", \"Self_Employed\", \"Education\", \"Property_Area\"],axis=1, inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 102,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 311,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Gender</th>\n",
       "      <th>Married</th>\n",
       "      <th>Dependents</th>\n",
       "      <th>Education</th>\n",
       "      <th>Self_Employed</th>\n",
       "      <th>ApplicantIncome</th>\n",
       "      <th>CoapplicantIncome</th>\n",
       "      <th>LoanAmount</th>\n",
       "      <th>Loan_Amount_Term</th>\n",
       "      <th>Credit_History</th>\n",
       "      <th>Property_Area</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Male</td>\n",
       "      <td>No</td>\n",
       "      <td>0</td>\n",
       "      <td>Not Graduate</td>\n",
       "      <td>No</td>\n",
       "      <td>3748</td>\n",
       "      <td>1668.0</td>\n",
       "      <td>110.0</td>\n",
       "      <td>360.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>Semiurban</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Male</td>\n",
       "      <td>Yes</td>\n",
       "      <td>3</td>\n",
       "      <td>Graduate</td>\n",
       "      <td>No</td>\n",
       "      <td>4000</td>\n",
       "      <td>7750.0</td>\n",
       "      <td>290.0</td>\n",
       "      <td>360.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>Semiurban</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Male</td>\n",
       "      <td>Yes</td>\n",
       "      <td>0</td>\n",
       "      <td>Graduate</td>\n",
       "      <td>No</td>\n",
       "      <td>2625</td>\n",
       "      <td>6250.0</td>\n",
       "      <td>187.0</td>\n",
       "      <td>360.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>Rural</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Male</td>\n",
       "      <td>No</td>\n",
       "      <td>0</td>\n",
       "      <td>Not Graduate</td>\n",
       "      <td>No</td>\n",
       "      <td>3902</td>\n",
       "      <td>1666.0</td>\n",
       "      <td>109.0</td>\n",
       "      <td>360.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>Rural</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Male</td>\n",
       "      <td>Yes</td>\n",
       "      <td>0</td>\n",
       "      <td>Not Graduate</td>\n",
       "      <td>No</td>\n",
       "      <td>6096</td>\n",
       "      <td>0.0</td>\n",
       "      <td>218.0</td>\n",
       "      <td>360.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>Rural</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "  Gender Married  Dependents     Education Self_Employed  ApplicantIncome  \\\n",
       "0   Male      No           0  Not Graduate            No             3748   \n",
       "1   Male     Yes           3      Graduate            No             4000   \n",
       "2   Male     Yes           0      Graduate            No             2625   \n",
       "3   Male      No           0  Not Graduate            No             3902   \n",
       "4   Male     Yes           0  Not Graduate            No             6096   \n",
       "\n",
       "   CoapplicantIncome  LoanAmount  Loan_Amount_Term  Credit_History  \\\n",
       "0             1668.0       110.0             360.0             1.0   \n",
       "1             7750.0       290.0             360.0             1.0   \n",
       "2             6250.0       187.0             360.0             1.0   \n",
       "3             1666.0       109.0             360.0             1.0   \n",
       "4                0.0       218.0             360.0             0.0   \n",
       "\n",
       "  Property_Area  \n",
       "0     Semiurban  \n",
       "1     Semiurban  \n",
       "2         Rural  \n",
       "3         Rural  \n",
       "4         Rural  "
      ]
     },
     "execution_count": 311,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "test.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 103,
   "metadata": {},
   "outputs": [],
   "source": [
    "test=pd.get_dummies(test, drop_first=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 104,
   "metadata": {},
   "outputs": [],
   "source": [
    "#test_s=sc.transform(test)\n",
    "test_r=rfe.transform(test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 108,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>0</th>\n",
       "      <th>1</th>\n",
       "      <th>2</th>\n",
       "      <th>3</th>\n",
       "      <th>4</th>\n",
       "      <th>5</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0.0</td>\n",
       "      <td>3748.0</td>\n",
       "      <td>1668.0</td>\n",
       "      <td>110.0</td>\n",
       "      <td>360.0</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>3.0</td>\n",
       "      <td>4000.0</td>\n",
       "      <td>7750.0</td>\n",
       "      <td>290.0</td>\n",
       "      <td>360.0</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>0.0</td>\n",
       "      <td>2625.0</td>\n",
       "      <td>6250.0</td>\n",
       "      <td>187.0</td>\n",
       "      <td>360.0</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>0.0</td>\n",
       "      <td>3902.0</td>\n",
       "      <td>1666.0</td>\n",
       "      <td>109.0</td>\n",
       "      <td>360.0</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0.0</td>\n",
       "      <td>6096.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>218.0</td>\n",
       "      <td>360.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>118</th>\n",
       "      <td>0.0</td>\n",
       "      <td>4683.0</td>\n",
       "      <td>1915.0</td>\n",
       "      <td>185.0</td>\n",
       "      <td>360.0</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>119</th>\n",
       "      <td>2.0</td>\n",
       "      <td>3601.0</td>\n",
       "      <td>1590.0</td>\n",
       "      <td>131.0</td>\n",
       "      <td>360.0</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>120</th>\n",
       "      <td>0.0</td>\n",
       "      <td>3017.0</td>\n",
       "      <td>663.0</td>\n",
       "      <td>102.0</td>\n",
       "      <td>360.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>121</th>\n",
       "      <td>0.0</td>\n",
       "      <td>17263.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>225.0</td>\n",
       "      <td>360.0</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>122</th>\n",
       "      <td>1.0</td>\n",
       "      <td>3750.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>116.0</td>\n",
       "      <td>360.0</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>123 rows × 6 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "       0        1       2      3      4    5\n",
       "0    0.0   3748.0  1668.0  110.0  360.0  1.0\n",
       "1    3.0   4000.0  7750.0  290.0  360.0  1.0\n",
       "2    0.0   2625.0  6250.0  187.0  360.0  1.0\n",
       "3    0.0   3902.0  1666.0  109.0  360.0  1.0\n",
       "4    0.0   6096.0     0.0  218.0  360.0  0.0\n",
       "..   ...      ...     ...    ...    ...  ...\n",
       "118  0.0   4683.0  1915.0  185.0  360.0  1.0\n",
       "119  2.0   3601.0  1590.0  131.0  360.0  1.0\n",
       "120  0.0   3017.0   663.0  102.0  360.0  0.0\n",
       "121  0.0  17263.0     0.0  225.0  360.0  1.0\n",
       "122  1.0   3750.0     0.0  116.0  360.0  1.0\n",
       "\n",
       "[123 rows x 6 columns]"
      ]
     },
     "execution_count": 108,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "finaldata=pd.DataFrame(test_r)\n",
    "finaldata"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 253,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Feature</th>\n",
       "      <th>Ranking</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Dependents</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>ApplicantIncome</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>CoapplicantIncome</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>LoanAmount</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Loan_Amount_Term</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>Credit_History</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>10</th>\n",
       "      <td>Property_Area_Semiurban</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>Education_Not Graduate</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>Married_Yes</td>\n",
       "      <td>4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>Gender_Male</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>11</th>\n",
       "      <td>Property_Area_Urban</td>\n",
       "      <td>6</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>Self_Employed_Yes</td>\n",
       "      <td>7</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                    Feature  Ranking\n",
       "0                Dependents        1\n",
       "1           ApplicantIncome        1\n",
       "2         CoapplicantIncome        1\n",
       "3                LoanAmount        1\n",
       "4          Loan_Amount_Term        1\n",
       "5            Credit_History        1\n",
       "10  Property_Area_Semiurban        2\n",
       "8    Education_Not Graduate        3\n",
       "7               Married_Yes        4\n",
       "6               Gender_Male        5\n",
       "11      Property_Area_Urban        6\n",
       "9         Self_Employed_Yes        7"
      ]
     },
     "execution_count": 253,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "selected_rfe_features = pd.DataFrame({'Feature':list(X_train.columns),\n",
    "                                      'Ranking':rfe.ranking_})\n",
    "selected_rfe_features.sort_values(by='Ranking')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 254,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Dependents</th>\n",
       "      <th>ApplicantIncome</th>\n",
       "      <th>CoapplicantIncome</th>\n",
       "      <th>LoanAmount</th>\n",
       "      <th>Loan_Amount_Term</th>\n",
       "      <th>Credit_History</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>-0.745545</td>\n",
       "      <td>-0.258318</td>\n",
       "      <td>0.104546</td>\n",
       "      <td>-1.688653</td>\n",
       "      <td>0.268498</td>\n",
       "      <td>0.403376</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2.239535</td>\n",
       "      <td>-0.221447</td>\n",
       "      <td>3.391888</td>\n",
       "      <td>-1.667964</td>\n",
       "      <td>0.268498</td>\n",
       "      <td>0.403376</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>-0.745545</td>\n",
       "      <td>-0.422628</td>\n",
       "      <td>2.581133</td>\n",
       "      <td>-1.679802</td>\n",
       "      <td>0.268498</td>\n",
       "      <td>0.403376</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>-0.745545</td>\n",
       "      <td>-0.235785</td>\n",
       "      <td>0.103465</td>\n",
       "      <td>-1.688767</td>\n",
       "      <td>0.268498</td>\n",
       "      <td>0.403376</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>-0.745545</td>\n",
       "      <td>0.085227</td>\n",
       "      <td>-0.797014</td>\n",
       "      <td>-1.676239</td>\n",
       "      <td>0.268498</td>\n",
       "      <td>-2.479079</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Dependents  ApplicantIncome  CoapplicantIncome  LoanAmount  \\\n",
       "0   -0.745545        -0.258318           0.104546   -1.688653   \n",
       "1    2.239535        -0.221447           3.391888   -1.667964   \n",
       "2   -0.745545        -0.422628           2.581133   -1.679802   \n",
       "3   -0.745545        -0.235785           0.103465   -1.688767   \n",
       "4   -0.745545         0.085227          -0.797014   -1.676239   \n",
       "\n",
       "   Loan_Amount_Term  Credit_History  \n",
       "0          0.268498        0.403376  \n",
       "1          0.268498        0.403376  \n",
       "2          0.268498        0.403376  \n",
       "3          0.268498        0.403376  \n",
       "4          0.268498       -2.479079  "
      ]
     },
     "execution_count": 254,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "finaldf=pd.DataFrame(test_r, columns=[\"Dependents\", \"ApplicantIncome\", \"CoapplicantIncome\",\"LoanAmount\", \"Loan_Amount_Term\", \"Credit_History\" ])\n",
    "finaldf.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 109,
   "metadata": {},
   "outputs": [],
   "source": [
    "final1=tpot.predict(test_r)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 112,
   "metadata": {},
   "outputs": [],
   "source": [
    "final2=rf.predict(test_r)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 113,
   "metadata": {},
   "outputs": [],
   "source": [
    "final3=logreg.predict(test_r)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 114,
   "metadata": {},
   "outputs": [],
   "source": [
    "submission1=pd.DataFrame(final1, columns=[\"prediction\"])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 115,
   "metadata": {},
   "outputs": [],
   "source": [
    "submission2=pd.DataFrame(final2, columns=[\"prediction\"])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 116,
   "metadata": {},
   "outputs": [],
   "source": [
    "submission3=pd.DataFrame(final3, columns=[\"prediction\"])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 243,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>prediction</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   prediction\n",
       "0           1\n",
       "1           1\n",
       "2           1\n",
       "3           1\n",
       "4           0"
      ]
     },
     "execution_count": 243,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "submission2.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 117,
   "metadata": {},
   "outputs": [],
   "source": [
    "submission1.to_csv(\"f21.csv\", index=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 118,
   "metadata": {},
   "outputs": [],
   "source": [
    "submission3.to_csv(\"f22.csv\", index=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 119,
   "metadata": {},
   "outputs": [],
   "source": [
    "submission2.to_csv(\"f23.csv\", index=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 120,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pickle\n",
    "pickle.dump(logreg, open(\"model.pkl\", \"wb\"))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 128,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[1]\n"
     ]
    }
   ],
   "source": [
    "model=pickle.load(open(\"model.pkl\", \"rb\"))\n",
    "print(model.predict([[1, 5000, 0,300, 240, 1]]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 127,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[0]\n"
     ]
    }
   ],
   "source": [
    "print(model.predict([[1, 30000, 3000,5000, 360, 1]]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 269,
   "metadata": {},
   "outputs": [],
   "source": [
    "pickle.dump(rf, open(\"model3.pkl\", \"wb\"))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 272,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[1]\n"
     ]
    }
   ],
   "source": [
    "model=pickle.load(open(\"model3.pkl\", \"rb\"))\n",
    "print(model.predict([[0, 0, 0,0, 0, 0]]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
