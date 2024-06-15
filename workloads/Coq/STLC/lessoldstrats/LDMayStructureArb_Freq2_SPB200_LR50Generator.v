From QuickChick Require Import QuickChick. Import QcNotation.
From Coq Require Import Bool ZArith List. Import ListNotations.
From ExtLib Require Import Monad.
From ExtLib.Data.Monads Require Import OptionMonad.
Import MonadNotation.

From STLC Require Import Impl Spec.

Fixpoint genTyp (size : nat) (stack1 : nat) (stack2 : nat) : G (Typ) :=
  match size with
  | O  => 
    (* Frequency1 (single-branch) *) 
    (returnGen (TBool ))
  | S size1 => 
    (* Frequency2 *) (freq [
      (* TBool *) (match (size, stack1, stack2) with
      | (1, 2, 1) => 50
      | (1, 2, 4) => 50
      | (2, 0, 2) => 50
      | (2, 3, 2) => 50
      | (2, 5, 2) => 50
      | (2, 6, 2) => 50
      | _ => 500
      end,
      (returnGen (TBool ))); 
      (* TFun *) (match (size, stack1, stack2) with
      | (1, 2, 1) => 50
      | (1, 2, 4) => 50
      | (2, 0, 2) => 50
      | (2, 3, 2) => 50
      | (2, 5, 2) => 50
      | (2, 6, 2) => 50
      | _ => 500
      end,
      (bindGen (genTyp size1 stack2 1) 
      (fun p1 => 
        (bindGen (genTyp size1 stack2 4) 
        (fun p2 => 
          (returnGen (TFun p1 p2)))))))])
  end.

Fixpoint genExpr (size : nat) (stack1 : nat) (stack2 : nat) : G (Expr) :=
  match size with
  | O  => 
    (* Frequency3 *) (freq [
      (* Var *) (match (size, stack1, stack2) with
      | (0, 3, 3) => 91
      | (0, 3, 5) => 87
      | (0, 3, 6) => 93
      | (0, 5, 3) => 89
      | (0, 5, 5) => 68
      | (0, 5, 6) => 61
      | (0, 6, 3) => 90
      | (0, 6, 5) => 90
      | (0, 6, 6) => 92
      | _ => 500
      end,
      (bindGen 
      arbitrary 
      (fun p1 => 
        (returnGen (Var p1))))); 
      (* Bool *) (match (size, stack1, stack2) with
      | (0, 3, 3) => 0
      | (0, 3, 5) => 90
      | (0, 3, 6) => 77
      | (0, 5, 3) => 0
      | (0, 5, 5) => 73
      | (0, 5, 6) => 76
      | (0, 6, 3) => 0
      | (0, 6, 5) => 84
      | (0, 6, 6) => 77
      | _ => 500
      end,
      (bindGen 
      arbitrary 
      (fun p1 => 
        (returnGen (Bool p1)))))])
  | S size1 => 
    (* Frequency4 *) (freq [
      (* Var *) (match (size, stack1, stack2) with
      | (1, 3, 3) => 61
      | (1, 3, 5) => 41
      | (1, 3, 6) => 21
      | (1, 5, 3) => 47
      | (1, 5, 5) => 41
      | (1, 5, 6) => 78
      | (1, 6, 3) => 35
      | (1, 6, 5) => 58
      | (1, 6, 6) => 40
      | (2, 3, 3) => 1
      | (2, 3, 5) => 7
      | (2, 3, 6) => 1
      | (2, 5, 3) => 57
      | (2, 5, 5) => 36
      | (2, 5, 6) => 55
      | (2, 6, 3) => 1
      | (2, 6, 5) => 3
      | (2, 6, 6) => 3
      | (3, 3, 3) => 44
      | (3, 3, 5) => 46
      | (3, 3, 6) => 44
      | (3, 5, 3) => 0
      | (3, 5, 5) => 24
      | (3, 5, 6) => 0
      | (3, 6, 3) => 48
      | (3, 6, 5) => 54
      | (3, 6, 6) => 48
      | (4, 0, 3) => 16
      | (4, 0, 5) => 0
      | (4, 0, 6) => 17
      | (5, 0, 0) => 0
      | _ => 500
      end,
      (bindGen 
      arbitrary 
      (fun p1 => 
        (returnGen (Var p1))))); 
      (* Bool *) (match (size, stack1, stack2) with
      | (1, 3, 3) => 0
      | (1, 3, 5) => 58
      | (1, 3, 6) => 63
      | (1, 5, 3) => 1
      | (1, 5, 5) => 27
      | (1, 5, 6) => 21
      | (1, 6, 3) => 0
      | (1, 6, 5) => 41
      | (1, 6, 6) => 35
      | (2, 3, 3) => 0
      | (2, 3, 5) => 10
      | (2, 3, 6) => 0
      | (2, 5, 3) => 9
      | (2, 5, 5) => 40
      | (2, 5, 6) => 42
      | (2, 6, 3) => 0
      | (2, 6, 5) => 5
      | (2, 6, 6) => 3
      | (3, 3, 3) => 44
      | (3, 3, 5) => 47
      | (3, 3, 6) => 55
      | (3, 5, 3) => 0
      | (3, 5, 5) => 15
      | (3, 5, 6) => 0
      | (3, 6, 3) => 48
      | (3, 6, 5) => 49
      | (3, 6, 6) => 53
      | (4, 0, 3) => 16
      | (4, 0, 5) => 0
      | (4, 0, 6) => 63
      | (5, 0, 0) => 0
      | _ => 500
      end,
      (bindGen 
      arbitrary 
      (fun p1 => 
        (returnGen (Bool p1))))); 
      (* Abs *) (match (size, stack1, stack2) with
      | (1, 3, 3) => 90
      | (1, 3, 5) => 85
      | (1, 3, 6) => 94
      | (1, 5, 3) => 75
      | (1, 5, 5) => 60
      | (1, 5, 6) => 49
      | (1, 6, 3) => 91
      | (1, 6, 5) => 82
      | (1, 6, 6) => 93
      | (2, 3, 3) => 29
      | (2, 3, 5) => 18
      | (2, 3, 6) => 10
      | (2, 5, 3) => 63
      | (2, 5, 5) => 60
      | (2, 5, 6) => 58
      | (2, 6, 3) => 22
      | (2, 6, 5) => 40
      | (2, 6, 6) => 10
      | (3, 3, 3) => 60
      | (3, 3, 5) => 52
      | (3, 3, 6) => 53
      | (3, 5, 3) => 1
      | (3, 5, 5) => 66
      | (3, 5, 6) => 0
      | (3, 6, 3) => 53
      | (3, 6, 5) => 51
      | (3, 6, 6) => 51
      | (4, 0, 3) => 85
      | (4, 0, 5) => 0
      | (4, 0, 6) => 75
      | (5, 0, 0) => 97
      | _ => 500
      end,
      (bindGen (genTyp 2 stack2 2) 
      (fun p1 => 
        (bindGen (genExpr size1 stack2 5) 
        (fun p2 => 
          (returnGen (Abs p1 p2))))))); 
      (* App *) (match (size, stack1, stack2) with
      | (1, 3, 3) => 95
      | (1, 3, 5) => 79
      | (1, 3, 6) => 93
      | (1, 5, 3) => 80
      | (1, 5, 5) => 68
      | (1, 5, 6) => 42
      | (1, 6, 3) => 96
      | (1, 6, 5) => 79
      | (1, 6, 6) => 95
      | (2, 3, 3) => 95
      | (2, 3, 5) => 89
      | (2, 3, 6) => 95
      | (2, 5, 3) => 64
      | (2, 5, 5) => 60
      | (2, 5, 6) => 45
      | (2, 6, 3) => 96
      | (2, 6, 5) => 90
      | (2, 6, 6) => 94
      | (3, 3, 3) => 49
      | (3, 3, 5) => 55
      | (3, 3, 6) => 47
      | (3, 5, 3) => 96
      | (3, 5, 5) => 74
      | (3, 5, 6) => 96
      | (3, 6, 3) => 51
      | (3, 6, 5) => 46
      | (3, 6, 6) => 48
      | (4, 0, 3) => 29
      | (4, 0, 5) => 96
      | (4, 0, 6) => 23
      | (5, 0, 0) => 0
      | _ => 500
      end,
      (bindGen (genExpr size1 stack2 3) 
      (fun p1 => 
        (bindGen (genExpr size1 stack2 6) 
        (fun p2 => 
          (returnGen (App p1 p2)))))))])
  end.

Definition gSized :=
  (genExpr 5 0 0).

Definition test_prop_SinglePreserve :=
forAll gSized (fun (e: Expr) =>
  prop_SinglePreserve e).

(*! QuickChick test_prop_SinglePreserve. *)

Definition test_prop_MultiPreserve :=
forAll gSized (fun (e: Expr) =>
  prop_MultiPreserve e).

(*! QuickChick test_prop_MultiPreserve. *)
          
