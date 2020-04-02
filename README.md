# Invisble Seccurity Signature System
In biometrics, a person’s signature is considered one of the most reliable authentication modes.
Like in banks where high security applications are required, the use of signatures for real-time
and accurate personal authentication is necessary. However, it is also a non-negligible drawback
that stealers can imitate the signature by practice. In order to prevent the signature from being
stolen, we proposed the Invisible Signature Security Device (ISSD) to increase security.
Because each person's direction of signature trajectory in the air, the slope of the signature curve,
signature speed, acceleration, etc. are totally different, others basically cannot imitate the name
of the signer, thus ensuring safety and reliability. we also designed and implemented a friendly
GUI interface for human-computer interaction. The topic is based on Leap motion equipment
to complete high-precision tracking fingers for aerial signature, and to achieve 3D and invisible
signatures. After the features are extracted in the background platform, signatures will be
recognized by using Fast-DTW+K-NN algorithm, and the signature data will be compared with
the data in the local database to identify the signature. After experimental verification, the true
acceptance rate for single signature can reach 97%, and the false acceptance rate is
approximately 0.5%.

<div align=center>
  
<img src="https://github.com/fe1ixxu/ISSD/blob/master/demo/Issd_demo.gif" alt="demo" width="512px">
</div>

# Prerequisites
* Leap motion and its API
* required installation in ```requirement.txt```
* Platform Win-64

# Run

```
python leapinterface.py
```

# Built With
* [Leapmotion](https://developer.leapmotion.com) - The hardware framework
* [MySQL](https://www.mysql.com/) - Database
